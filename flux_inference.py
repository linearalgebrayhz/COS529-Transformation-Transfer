import argparse
import os
import torch
from PIL import Image
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from transformers import CLIPTokenizer, T5TokenizerFast
from transformers import CLIPTextModel, T5EncoderModel

from src.pipeline_pe_clone import FluxPipeline, position_encoding_clone

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for Flux model with LoRA weights")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        required=True,
        help="Path to the pretrained model directory",
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default=None,
        help="Path to the LoRA weights directory. If not provided, will use base model without LoRA.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--condition_image",
        type=str,
        required=True,
        help="Path to condition image",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Height of output image",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Width of output image",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Total number of images to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference (cuda, cpu)",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision type to use",
    )
    
    return parser.parse_args()

def load_pipeline(args):
    print(f"Loading model from {args.pretrained_model_path}")
    
    # Determine dtype based on mixed precision setting
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
        
    # Check if device supports bfloat16
    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError(
            "Mixed precision with bfloat16 is not supported on MPS. Please use fp16 or fp32 instead."
        )
        
    device = torch.device(args.device)
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_path,
        subfolder="vae",
        torch_dtype=weight_dtype
    )
    vae.to(device)
    
    # Load Transformer
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        device_map={"": device},
        low_cpu_mem_usage=True
    )
    
    # Load tokenizers and text encoders
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_path,
        subfolder="tokenizer",
    )
    
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_path,
        subfolder="tokenizer_2",
    )
    
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype
    )
    text_encoder_one.to(device)
    
    text_encoder_two = T5EncoderModel.from_pretrained(
        args.pretrained_model_path,
        subfolder="text_encoder_2",
        torch_dtype=weight_dtype
    )
    text_encoder_two.to(device)
    
    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_path, 
        subfolder="scheduler"
    )
    
    # Create pipeline
    pipeline = FluxPipeline.from_pretrained(
        args.pretrained_model_path,
        vae=vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=weight_dtype,
    )
    
    # Load LoRA weights if provided
    if args.lora_model_path:
        print(f"Loading LoRA weights from {args.lora_model_path}")
        pipeline.load_lora_weights(args.lora_model_path)
        # Optional: fuse the LoRA weights for faster inference
        # pipeline.fuse_lora() # Uncomment if you want to fuse LoRA weights
    
    pipeline.to(device)
    return pipeline

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        
    # Load pipeline
    pipeline = load_pipeline(args)
    
    # Load and resize condition image
    condition_image = Image.open(args.condition_image)
    condition_image = condition_image.resize((args.width, args.height), Image.LANCZOS)
    
    # Prepare generator
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
    
    # Set inference parameters
    pipeline.set_progress_bar_config(disable=False)
    
    # Calculate number of batches
    total_images = args.num_images
    full_batches = total_images // args.batch_size
    remainder = total_images % args.batch_size
    
    image_count = 0
    
    # Process full batches
    for batch_idx in range(full_batches):
        print(f"Generating batch {batch_idx+1}/{full_batches + (1 if remainder > 0 else 0)}")
        
        # Generate images
        outputs = pipeline(
            prompt=args.prompt,
            condition_image=condition_image,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
            generator=generator,
        )
        
        # Save images
        for image in outputs.images:
            image_path = os.path.join(args.output_dir, f"image_{image_count:05d}.png")
            image.save(image_path)
            print(f"Saved image to {image_path}")
            image_count += 1
    
    # Process remaining images (if any)
    if remainder > 0:
        print(f"Generating final batch of {remainder} images")
        
        outputs = pipeline(
            prompt=args.prompt,
            condition_image=condition_image,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=remainder,
            generator=generator,
        )
        
        # Save images
        for image in outputs.images:
            image_path = os.path.join(args.output_dir, f"image_{image_count:05d}.png")
            image.save(image_path)
            print(f"Saved image to {image_path}")
            image_count += 1
    
    print(f"Successfully generated {total_images} images!")

if __name__ == "__main__":
    main()