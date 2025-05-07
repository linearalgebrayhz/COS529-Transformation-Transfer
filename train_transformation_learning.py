import argparse
import copy
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
import re
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

from PIL import Image
from safetensors.torch import save_file

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
    convert_unet_state_dict_to_peft
)

from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

from src.pipeline_pe_clone import FluxPipeline, position_encoding_clone

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed
check_min_version("0.31.0.dev0")
logger = get_logger(__name__)

# Create a custom positional encoding for rotations
def rotation_position_encoding(batch_size, height, width, R_matrices, device, dtype=torch.float32):
    """
    Create positional encodings that incorporate rotation information.
    
    Args:
        batch_size: Number of images in batch
        height: Height of feature map
        width: Width of feature map
        R_matrices: Rotation matrices [batch_size, 3, 3]
        device: Device to create tensors on
        dtype: Data type for tensors
    """
    # Base positional encoding
    pos_encoding = position_encoding_clone(batch_size, height, width, device, dtype)
    
    # Extract rotation information (Euler angles)
    angles = torch.zeros(batch_size, 3, device=device, dtype=dtype)
    for i in range(batch_size):
        R = R_matrices[i]
        # Extract Euler angles from rotation matrix (approximate)
        angles[i, 0] = torch.atan2(R[2, 1], R[2, 2])  # Roll
        angles[i, 1] = torch.asin(-R[2, 0])           # Pitch
        angles[i, 2] = torch.atan2(R[1, 0], R[0, 0])  # Yaw
    
    # Create rotation feature channels
    rot_channels = 6  # sin and cos of each angle
    rot_encoding = torch.zeros(batch_size, rot_channels, height, width, device=device, dtype=dtype)
    
    # Apply rotation information across spatial dimensions
    for i in range(batch_size):
        # Sin and cos of each angle
        sin_angles = torch.sin(angles[i])
        cos_angles = torch.cos(angles[i])
        
        # Fill channels with rotation information
        for h in range(height):
            for w in range(width):
                # First three channels: sin of angles
                rot_encoding[i, 0:3, h, w] = sin_angles
                # Next three channels: cos of angles
                rot_encoding[i, 3:6, h, w] = cos_angles
    
    # Concatenate with original positional encoding
    # Original shape: [batch_size, channels, height, width]
    # We add the rotation channels
    combined_encoding = torch.cat([pos_encoding, rot_encoding], dim=1)
    
    return combined_encoding

# Encode transformation matrix as a feature vector
def encode_transformation_matrix(R_matrices, t_vectors, device, dtype=torch.float32):
    """
    Encode rotation matrices and translation vectors into feature vectors.
    
    Args:
        R_matrices: Batch of rotation matrices [batch_size, 3, 3]
        t_vectors: Batch of translation vectors [batch_size, 3]
        device: Device to create tensors on
        dtype: Data type for tensors
    
    Returns:
        Encoded transformation features [batch_size, feature_dim]
    """
    batch_size = R_matrices.shape[0]
    feature_dim = 12  # 9 for rotation + 3 for translation
    
    # Flatten each transformation into a feature vector
    features = torch.zeros(batch_size, feature_dim, device=device, dtype=dtype)
    
    for i in range(batch_size):
        # Flatten rotation matrix (9 values)
        features[i, 0:9] = R_matrices[i].flatten()
        # Add translation vector (3 values)
        features[i, 9:12] = t_vectors[i]
    
    # Normalize features for better conditioning
    features = F.normalize(features, dim=1)
    
    return features

def log_validation(
        pipeline,
        args,
        accelerator,
        pipeline_args,
        step,
        torch_dtype,
        is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline.set_progress_bar_config(disable=True)
    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    autocast_ctx = nullcontext()

    with autocast_ctx:
        images = []
        for _ in range(args.num_validation_images):
            result = pipeline(**pipeline_args, generator=generator)
            images.append(result.images[0])

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, step, dataformats="NHWC")
            
            # Also log source and reference images
            if "condition_image" in pipeline_args:
                source_img = np.asarray(pipeline_args["condition_image"])
                tracker.writer.add_image(f"{phase_name}_source", source_img, step, dataformats="HWC")
            
            if "reference_image" in pipeline_args:
                ref_img = np.asarray(pipeline_args["reference_image"])
                tracker.writer.add_image(f"{phase_name}_reference", ref_img, step, dataformats="HWC")
                
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )
            
            # Also log source and reference images
            if "condition_image" in pipeline_args:
                tracker.log({f"{phase_name}_source": wandb.Image(pipeline_args["condition_image"], caption="Source")})
            
            if "reference_image" in pipeline_args:
                tracker.log({f"{phase_name}_reference": wandb.Image(pipeline_args["reference_image"], caption="Reference")})

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training script for transformation learning.")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `meta.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--first_stage_model_path",
        type=str,
        default=None,
        help="Path to the first stage model (optional). If provided, will be used to initialize the model.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--source_column",
        type=str,
        default="source",
        help="The column of the dataset containing the source image.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="target",
        help="The column of the dataset containing the target image.",
    )
    parser.add_argument(
        "--test_source_column",
        type=str,
        default="test_source",
        help="The column of the dataset containing the test source image.",
    )
    parser.add_argument(
        "--relative_transform_column",
        type=str,
        default="relative_transform",
        help="The column containing relative transformation matrices.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing the prompt for each image",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="direct",
        choices=["direct", "matrix_feature", "rotational_encoding"],
        help="Method to use for transformation learning.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="Transform the image",
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_source_image",
        type=str,
        default=None,
        help="Source image to use for validation.",
    )
    parser.add_argument(
        "--validation_reference_source",
        type=str,
        default=None,
        help="Reference source image to use for validation.",
    )
    parser.add_argument(
        "--validation_reference_target",
        type=str,
        default=None,
        help="Reference target image to use for validation.",
    )
    parser.add_argument(
        "--validation_transform",
        type=str,
        default=None,
        help="JSON string of transformation matrix to use for validation.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help=(
            "Run validation every X steps. Validation consists of generating images from a fixed set of prompts and"
            " calculating metrics for them."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./transformation_model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--height",
        type=int,
        default=576,
        help=(
            "The height for input images, all the images in the train/validation dataset will be resized to this"
            " height"
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help=(
            "The width for input images, all the images in the train/validation dataset will be resized to this"
            " width"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1,
        help="Guidance scale for training",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for transformer params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer.",
    )

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


# Custom dataset for transformation learning
class TransformationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        json_file,
        tokenizers,
        size=(576, 768),
        interpolation="bicubic",
        method="direct",
    ):
        self.data_root = data_root
        self.tokenizers = tokenizers
        self.size = size
        self.method = method
        
        # Load the dataset
        import json
        with open(json_file, 'r') as f:
            self.dataset = [json.loads(line) for line in f]
            
        self.interpolation = {
            "linear": Image.BILINEAR,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }[interpolation]
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Load source, target, and test_source images
        source_path = os.path.join(self.data_root, item["source"])
        target_path = os.path.join(self.data_root, item["target"])
        test_source_path = os.path.join(self.data_root, item["test_source"])
        
        source_image = Image.open(source_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")
        test_source_image = Image.open(test_source_path).convert("RGB")
        
        # Resize images
        source_image = source_image.resize(self.size, resample=self.interpolation)
        target_image = target_image.resize(self.size, resample=self.interpolation)
        test_source_image = test_source_image.resize(self.size, resample=self.interpolation)
        
        # Convert to tensors
        source_tensor = torch.from_numpy(np.array(source_image)).float() / 127.5 - 1.0
        target_tensor = torch.from_numpy(np.array(target_image)).float() / 127.5 - 1.0
        test_source_tensor = torch.from_numpy(np.array(test_source_image)).float() / 127.5 - 1.0
        
        # Rearrange from HWC to CHW
        source_tensor = source_tensor.permute(2, 0, 1)
        target_tensor = target_tensor.permute(2, 0, 1)
        test_source_tensor = test_source_tensor.permute(2, 0, 1)
        
        # Get transformation matrix if available
        if "relative_transform" in item:
            # Extract transformation matrices
            R = torch.tensor(item["relative_transform"]["R"], dtype=torch.float32)
            t = torch.tensor(item["relative_transform"]["t"], dtype=torch.float32)
        else:
            # Default identity transformation if not provided
            R = torch.eye(3, dtype=torch.float32)
            t = torch.zeros(3, dtype=torch.float32)
        
        # Get caption (or use default)
        caption = item.get("caption", "Transform the image")
        
        # Tokenize text
        tokenized_text_1 = self.tokenizers[0](
            caption,
            padding="max_length",
            max_length=self.tokenizers[0].model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokenized_text_2 = self.tokenizers[1](
            caption,
            padding="max_length",
            max_length=self.tokenizers[1].model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "text_ids_1": tokenized_text_1.input_ids[0],
            "text_ids_2": tokenized_text_2.input_ids[0],
            "source_pixel_values": source_tensor,    # I_source
            "target_pixel_values": target_tensor,    # I_target
            "test_source_pixel_values": test_source_tensor,  # T_source
            "R_matrix": R,  # Rotation matrix
            "t_vector": t,  # Translation vector
        }


def collate_fn(examples):
    text_ids_1 = torch.stack([example["text_ids_1"] for example in examples])
    text_ids_2 = torch.stack([example["text_ids_2"] for example in examples])
    source_pixel_values = torch.stack([example["source_pixel_values"] for example in examples])
    target_pixel_values = torch.stack([example["target_pixel_values"] for example in examples])
    test_source_pixel_values = torch.stack([example["test_source_pixel_values"] for example in examples])
    R_matrices = torch.stack([example["R_matrix"] for example in examples])
    t_vectors = torch.stack([example["t_vector"] for example in examples])
    
    return {
        "text_ids_1": text_ids_1,
        "text_ids_2": text_ids_2,
        "source_pixel_values": source_pixel_values,
        "target_pixel_values": target_pixel_values,
        "test_source_pixel_values": test_source_pixel_values,
        "R_matrices": R_matrices,
        "t_vectors": t_vectors,
    }


def encode_token_ids(text_encoders, tokens, accelerator):
    """Encode the token ids using the text encoders."""
    text_encoder_one, text_encoder_two = text_encoders
    
    with torch.no_grad():
        # Process with CLIP text encoder
        prompt_embeds_one = text_encoder_one(
            tokens[0].to(text_encoder_one.device),
            output_hidden_states=True,
        )
        prompt_embeds_one_hidden_states = prompt_embeds_one.hidden_states[-2]
        
        # Process with T5 text encoder
        prompt_embeds_two = text_encoder_two(
            tokens[1].to(text_encoder_two.device),
            output_hidden_states=True,
        )
        prompt_embeds_two_hidden_states = prompt_embeds_two.hidden_states[-2]
        
        # Get pooled embeddings from T5
        pooled_prompt_embeds = prompt_embeds_two[0]
        
        # Handle concatenation properly
        # The issue is that CLIP has sequence length 77 and T5 has 512
        b, s_clip, d_clip = prompt_embeds_one_hidden_states.shape
        b, s_t5, d_t5 = prompt_embeds_two_hidden_states.shape
        
        # If sequence lengths don't match, we interpolate T5 to match CLIP
        if s_clip != s_t5:
            prompt_embeds_two_resized = torch.nn.functional.interpolate(
                prompt_embeds_two_hidden_states.permute(0, 2, 1),  # [b, d, s]
                size=s_clip,
                mode='linear'
            ).permute(0, 2, 1)  # [b, s_clip, d]
        else:
            prompt_embeds_two_resized = prompt_embeds_two_hidden_states
            
        # Now we can safely concatenate along the feature dimension
        prompt_embeds = torch.cat([prompt_embeds_one_hidden_states, prompt_embeds_two_resized], dim=-1)
        
    # Return text IDs from CLIP
    text_ids = tokens[0]
    
    return prompt_embeds, pooled_prompt_embeds, text_ids


# Custom FluxPipeline that supports transformation learning methods
class TransformationFluxPipeline(FluxPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt="",
        condition_image=None,
        reference_source=None,
        reference_target=None,
        transformation_matrix=None,
        height=576,
        width=768,
        num_inference_steps=20,
        guidance_scale=1.0,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        max_sequence_length=512,
        output_type="pil",
        return_dict=True,
        callback=None,
        callback_steps=1,
        method="direct",
    ):
        # Process reference source and target images if provided
        if reference_source is not None and reference_target is not None:
            # Preprocess reference images
            reference_source_tensor = self.preprocess_image(reference_source)
            reference_target_tensor = self.preprocess_image(reference_target)
            
            # Extract the transformation from reference pair
            # This is a placeholder for the actual implementation
            # In a real implementation, you would extract the transformation
            # from the reference source and target
        
        # Process transformation matrix if provided
        if transformation_matrix is not None:
            R_matrix = torch.tensor(transformation_matrix["R"], device=self.device)
            t_vector = torch.tensor(transformation_matrix["t"], device=self.device)
        else:
            # Default identity transformation
            R_matrix = torch.eye(3, device=self.device)
            t_vector = torch.zeros(3, device=self.device)
            
        # Handle different methods
        if method == "direct":
            # For direct method, we don't need to do anything special
            # Just use the standard pipeline
            return super().__call__(
                prompt=prompt,
                condition_image=condition_image,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta,
                generator=generator,
                max_sequence_length=max_sequence_length,
                output_type=output_type,
                return_dict=return_dict,
                callback=callback,
                callback_steps=callback_steps,
            )
        
        elif method == "matrix_feature":
            # Process condition image
            condition_image_tensor = self.preprocess_image(condition_image)
            
            # Encode the image
            image_latents = self.encode_image(condition_image_tensor)
            
            # Encode the transformation matrix as a feature
            transform_features = encode_transformation_matrix(
                R_matrix.unsqueeze(0), 
                t_vector.unsqueeze(0), 
                self.device
            )
            
            # TODO: Incorporate transformation features into the pipeline
            # This would involve modifying the _encode_prompt method to include
            # the transformation features
            
            # For now, we'll use the standard pipeline
            return super().__call__(
                prompt=prompt,
                condition_image=condition_image,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta,
                generator=generator,
                max_sequence_length=max_sequence_length,
                output_type=output_type,
                return_dict=return_dict,
                callback=callback,
                callback_steps=callback_steps,
            )
            
        elif method == "rotational_encoding":
            # Process condition image
            condition_image_tensor = self.preprocess_image(condition_image)
            
            # Encode the image
            image_latents = self.encode_image(condition_image_tensor)
            
            # TODO: Use rotation_position_encoding to create rotational position encodings
            # This would involve modifying the position encoding used in the pipeline
            
            # For now, we'll use the standard pipeline
            return super().__call__(
                prompt=prompt,
                condition_image=condition_image,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta,
                generator=generator,
                max_sequence_length=max_sequence_length,
                output_type=output_type,
                return_dict=return_dict,
                callback=callback,
                callback_steps=callback_steps,
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")


def main(args):
    # Basic setup
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. "
            "Please use fp16 (recommended) or fp32 instead."
        )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, args.logging_dir), exist_ok=True)
    logging_dir = Path(args.output_dir, args.logging_dir)

    # Setup accelerator
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # Set verbosity level
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    # Load tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # Import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder"
    )
    
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    # Set weight dtype based on mixed precision setting
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. "
            "Please use fp16 (recommended) or fp32 instead."
        )
    
    # Load models
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
    )
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    # If first stage model is provided, load its weights
    if args.first_stage_model_path is not None:
        logger.info(f"Loading weights from first stage model: {args.first_stage_model_path}")
        # Load transformer weights
        first_stage_transformer = FluxTransformer2DModel.from_pretrained(
            args.first_stage_model_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
        )
        # Copy weights
        transformer.load_state_dict(first_stage_transformer.state_dict(), strict=False)
        del first_stage_transformer
    
    # Setup gradient checkpointing
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    # For LoRA training, freeze base models
    transformer.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    vae.requires_grad_(False)
    
    # Move models to device
    text_encoder_one = text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two = text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    vae = vae.to(accelerator.device, dtype=weight_dtype)
    
    # Setup LoRA
    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
    
    # Configure LoRA
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)
    
    # Create dataset and dataloader
    tokenizers = [tokenizer_one, tokenizer_two]
    train_dataset = TransformationDataset(
        data_root=os.path.dirname(args.train_data_dir),
        json_file=args.train_data_dir,
        tokenizers=tokenizers,
        size=(args.height, args.width),
        method=args.method,
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    
    # Get VAE scaling factors
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    
    # Create a copy of the scheduler for the training loop
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    # Setup optimizer
    params_to_optimize = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        
    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    # Prepare for training with accelerator
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    # Initialize accelerator trackers
    if accelerator.is_main_process:
        tracker_name = "flux-transformation"
        accelerator.init_trackers(tracker_name, config=vars(args))
    
    # Print training info
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Using method: {args.method}")
    
    # Resume from checkpoint if specified
    global_step = 0
    first_epoch = 0
    
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        if path == "latest":
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            
        if path is not None:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
    
    # Helper function to get sigmas for diffusion
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    # Progress bar
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    # Training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Process text tokens
                tokens = [batch["text_ids_1"], batch["text_ids_2"]]
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_token_ids(
                    [text_encoder_one, text_encoder_two], tokens, accelerator
                )
                prompt_embeds = prompt_embeds.to(dtype=vae.dtype, device=accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=vae.dtype, device=accelerator.device)
                text_ids = text_ids.to(dtype=vae.dtype, device=accelerator.device)

                # Process images
                source_pixel_values = batch["source_pixel_values"].to(dtype=vae.dtype)  # I_source
                target_pixel_values = batch["target_pixel_values"].to(dtype=vae.dtype)  # I_target
                test_source_pixel_values = batch["test_source_pixel_values"].to(dtype=vae.dtype)  # T_source
                
                # Get transformation matrices
                R_matrices = batch["R_matrices"].to(device=accelerator.device, dtype=vae.dtype)
                t_vectors = batch["t_vectors"].to(device=accelerator.device, dtype=vae.dtype)
                
                # Encode images with VAE
                with torch.no_grad():
                    # Encode target image (I_target) - this is what we want to predict
                    target_latents = vae.encode(target_pixel_values).latent_dist.sample()
                    target_latents = (target_latents - vae_config_shift_factor) * vae_config_scaling_factor
                    target_latents = target_latents.to(dtype=weight_dtype)
                    
                    # Encode source image (I_source) - this is one of the conditioning inputs
                    source_latents = vae.encode(source_pixel_values).latent_dist.sample()
                    source_latents = (source_latents - vae_config_shift_factor) * vae_config_scaling_factor
                    source_latents = source_latents.to(dtype=weight_dtype)
                    
                    # Encode test source image (T_source) - this is another conditioning input
                    test_source_latents = vae.encode(test_source_pixel_values).latent_dist.sample()
                    test_source_latents = (test_source_latents - vae_config_shift_factor) * vae_config_scaling_factor
                    test_source_latents = test_source_latents.to(dtype=weight_dtype)
                
                # Sample noise for training
                noise = torch.randn_like(target_latents)
                bsz = target_latents.shape[0]
                
                # Sample timesteps for diffusion
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=target_latents.device)
                
                # Add noise according to flow matching
                sigmas = get_sigmas(timesteps, n_dim=target_latents.ndim, dtype=target_latents.dtype)
                noisy_target_latents = (1.0 - sigmas) * target_latents + sigmas * noise
                
                # Prepare position encoding
                vae_scale_factor = 16
                height_ = 2 * (int(source_pixel_values.shape[-2]) // vae_scale_factor)
                width_ = 2 * (int(source_pixel_values.shape[-1]) // vae_scale_factor)
                
                # Different conditioning based on the method
                if args.method == "direct":
                    # Direct method uses default position encoding
                    latent_image_ids = position_encoding_clone(
                        source_latents.shape[0],
                        height_,
                        width_,
                        accelerator.device,
                        weight_dtype,
                    )
                    
                elif args.method == "matrix_feature":
                    # Matrix feature method uses default position encoding
                    # but will add transformation features later
                    latent_image_ids = position_encoding_clone(
                        source_latents.shape[0],
                        height_,
                        width_,
                        accelerator.device,
                        weight_dtype,
                    )
                    
                    # Create transformation features
                    transform_features = encode_transformation_matrix(
                        R_matrices, t_vectors, accelerator.device, weight_dtype
                    )
                    
                    # Add transformation features to pooled prompt embeddings
                    # This is one way to incorporate the transformation information
                    # Another approach would be to add a new input to the transformer model
                    pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, transform_features], dim=1)
                    
                elif args.method == "rotational_encoding":
                    # Rotational encoding method uses custom position encoding
                    latent_image_ids = rotation_position_encoding(
                        source_latents.shape[0],
                        height_,
                        width_,
                        R_matrices,
                        accelerator.device,
                        weight_dtype,
                    )
                
                # Pack latents for the transformer model
                # For target prediction, we use I_source and T_source as conditioning inputs
                # and predict T_target (which should match I_target in the training data)
                
                # First, pack the noisy target latents (these are the ones the model learns to denoise)
                packed_noisy_target_latents = FluxPipeline._pack_latents(
                    noisy_target_latents,
                    batch_size=target_latents.shape[0],
                    num_channels_latents=target_latents.shape[1],
                    height=target_latents.shape[2],
                    width=target_latents.shape[3],
                )
                
                # Pack the source latents
                packed_source_latents = FluxPipeline._pack_latents(
                    source_latents,
                    batch_size=source_latents.shape[0],
                    num_channels_latents=source_latents.shape[1],
                    height=source_latents.shape[2],
                    width=source_latents.shape[3],
                )
                
                # Pack the test source latents
                packed_test_source_latents = FluxPipeline._pack_latents(
                    test_source_latents,
                    batch_size=test_source_latents.shape[0],
                    num_channels_latents=test_source_latents.shape[1],
                    height=test_source_latents.shape[2],
                    width=test_source_latents.shape[3],
                )
                
                # Concatenate all packed latents
                # The model will use packed_source_latents and packed_test_source_latents as conditioning
                # to predict packed_noisy_target_latents
                packed_latents = torch.concat([
                    packed_noisy_target_latents,  # First: what we're trying to predict (T_target)
                    packed_source_latents,        # Second: source conditioning (I_source)
                    packed_test_source_latents    # Third: test source conditioning (T_source)
                ], dim=-2)
                
                # Handle guidance
                if transformer.config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(target_latents.shape[0])
                else:
                    guidance = None
                
                # Forward pass through the transformer model
                model_pred = transformer(
                    hidden_states=packed_latents,
                    timestep=timesteps / 1000,  # Scale timesteps as in original code
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                
                # Extract only the prediction part (not the conditioning parts)
                model_pred = model_pred[:, :packed_noisy_target_latents.shape[1], :]
                
                # Unpack the predicted latents
                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=int(source_pixel_values.shape[-2]),
                    width=int(source_pixel_values.shape[-1]),
                    vae_scale_factor=vae_scale_factor,
                )
                
                # Apply weighting scheme
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                
                # Calculate target (flow matching loss)
                target = noise - target_latents
                
                # Compute the loss
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                
                # Backpropagate
                accelerator.backward(loss)
                
                # Clip gradients
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                
                # Update weights
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Save checkpoint
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    
                    # Clean up old checkpoints
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        
                        # Remove oldest checkpoints if we exceed the limit
                        if len(checkpoints) > args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                            removing_checkpoints = checkpoints[:num_to_remove]
                            
                            for checkpoint_to_remove in removing_checkpoints:
                                remove_path = os.path.join(args.output_dir, checkpoint_to_remove)
                                shutil.rmtree(remove_path)
                                logger.info(f"Removed checkpoint: {remove_path}")
            
            # Log metrics
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            # Run validation
            if accelerator.is_main_process and args.validation_source_image is not None and global_step % args.validation_steps == 0:
                # Load validation images
                validation_source = Image.open(args.validation_source_image).convert("RGB")
                validation_source = validation_source.resize((args.width, args.height), Image.LANCZOS)
                
                validation_ref_source = None
                validation_ref_target = None
                if args.validation_reference_source is not None and args.validation_reference_target is not None:
                    validation_ref_source = Image.open(args.validation_reference_source).convert("RGB")
                    validation_ref_source = validation_ref_source.resize((args.width, args.height), Image.LANCZOS)
                    
                    validation_ref_target = Image.open(args.validation_reference_target).convert("RGB")
                    validation_ref_target = validation_ref_target.resize((args.width, args.height), Image.LANCZOS)
                
                # Parse transformation matrix if provided
                transform_matrix = None
                if args.validation_transform is not None:
                    import json
                    transform_matrix = json.loads(args.validation_transform)
                
                # Create custom pipeline for validation
                pipeline = TransformationFluxPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    text_encoder=accelerator.unwrap_model(text_encoder_one),
                    text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                    transformer=accelerator.unwrap_model(transformer),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                
                # Setup pipeline arguments
                pipeline_args = {
                    "prompt": args.validation_prompt,
                    "condition_image": validation_source,
                    "reference_source": validation_ref_source,
                    "reference_target": validation_ref_target,
                    "transformation_matrix": transform_matrix,
                    "height": args.height,
                    "width": args.width,
                    "guidance_scale": 3.5,
                    "num_inference_steps": 20,
                    "max_sequence_length": 512,
                    "method": args.method,
                }
                
                # Run validation
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=pipeline_args,
                    step=global_step,
                    torch_dtype=weight_dtype,
                )
                
                # Save validation images
                save_path = os.path.join(args.output_dir, "validation")
                os.makedirs(save_path, exist_ok=True)
                save_folder = os.path.join(save_path, f"checkpoint-{global_step}")
                os.makedirs(save_folder, exist_ok=True)
                
                # Save source image for reference
                validation_source.save(os.path.join(save_folder, "source.jpg"))
                
                # Save reference images if available
                if validation_ref_source is not None:
                    validation_ref_source.save(os.path.join(save_folder, "ref_source.jpg"))
                if validation_ref_target is not None:
                    validation_ref_target.save(os.path.join(save_folder, "ref_target.jpg"))
                
                # Save generated images
                for idx, img in enumerate(images):
                    img.save(os.path.join(save_folder, f"generated_{idx}.jpg"))
                
                del pipeline
                torch.cuda.empty_cache()
            
            # Check if we've reached the end of training
            if global_step >= args.max_train_steps:
                break
    
    # Save the final model
    accelerator.wait_for_everyone()
    
    # Unwrap the transformed model
    transformer = accelerator.unwrap_model(transformer)
    
    if accelerator.is_main_process:
        # Save the trained LoRA weights
        save_path = os.path.join(args.output_dir, "lora_weights")
        os.makedirs(save_path, exist_ok=True)
        
        lora_state_dict = get_peft_model_state_dict(transformer)
        torch.save(lora_state_dict, os.path.join(save_path, f"pytorch_lora_weights.bin"))
        
        # Also save in safetensors format
        save_file(lora_state_dict, os.path.join(save_path, f"lora_weights.safetensors"))
        
        # Save the configuration
        transformer_lora_config.save_pretrained(save_path)
        
        # Create the full pipeline for inference
        pipeline = TransformationFluxPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            transformer=transformer,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        
        # Save metadata about the training
        import json
        metadata = {
            "method": args.method,
            "training_args": vars(args),
            "model_type": "transformation_learning",
        }
        with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Run a final validation
        if args.validation_source_image is not None:
            # Load validation images
            validation_source = Image.open(args.validation_source_image).convert("RGB")
            validation_source = validation_source.resize((args.width, args.height), Image.LANCZOS)
            
            validation_ref_source = None
            validation_ref_target = None
            if args.validation_reference_source is not None and args.validation_reference_target is not None:
                validation_ref_source = Image.open(args.validation_reference_source).convert("RGB")
                validation_ref_source = validation_ref_source.resize((args.width, args.height), Image.LANCZOS)
                
                validation_ref_target = Image.open(args.validation_reference_target).convert("RGB")
                validation_ref_target = validation_ref_target.resize((args.width, args.height), Image.LANCZOS)
            
            # Parse transformation matrix if provided
            transform_matrix = None
            if args.validation_transform is not None:
                import json
                transform_matrix = json.loads(args.validation_transform)
            
            # Setup pipeline arguments
            pipeline_args = {
                "prompt": args.validation_prompt,
                "condition_image": validation_source,
                "reference_source": validation_ref_source,
                "reference_target": validation_ref_target,
                "transformation_matrix": transform_matrix,
                "height": args.height,
                "width": args.width,
                "guidance_scale": 3.5,
                "num_inference_steps": 20,
                "max_sequence_length": 512,
                "method": args.method,
            }
            
            # Run final validation
            images = log_validation(
                pipeline=pipeline,
                args=args,
                accelerator=accelerator,
                pipeline_args=pipeline_args,
                step=global_step,
                torch_dtype=weight_dtype,
                is_final_validation=True,
            )
            
            # Save validation images
            save_path = os.path.join(args.output_dir, "final_validation")
            os.makedirs(save_path, exist_ok=True)
            
            # Save source image for reference
            validation_source.save(os.path.join(save_path, "source.jpg"))
            
            # Save reference images if available
            if validation_ref_source is not None:
                validation_ref_source.save(os.path.join(save_path, "ref_source.jpg"))
            if validation_ref_target is not None:
                validation_ref_target.save(os.path.join(save_path, "ref_target.jpg"))
            
            # Save generated images
            for idx, img in enumerate(images):
                img.save(os.path.join(save_path, f"generated_{idx}.jpg"))
    
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)