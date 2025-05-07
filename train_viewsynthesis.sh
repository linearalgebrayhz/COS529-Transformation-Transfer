#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Base configuration
export PRETRAINED_MODEL="black-forest-labs/FLUX.1-dev"  # Pretrained model from HuggingFace
export OUTPUT_DIR="outputs/viewsynthesis_model"
export CONFIG="/home/linux/.cache/huggingface/accelerate/default_config.yaml"  # Change this to your accelerate config path
export TRAIN_DATA="data/BlendedMVS/meta.jsonl"  # Change this to your dataset path
export VALIDATION_IMAGE="val.jpg"  # Change this to your validation image
export LOG_PATH="$OUTPUT_DIR/logs"

# Create output directories if they don't exist
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_PATH

# Training command
accelerate launch --config_file $CONFIG train_flux_viewsynthesis.py \
    --pretrained_model_name_or_path $PRETRAINED_MODEL \
    --width 192 \
    --height 144 \
    --dataloader_num_workers 0 \
    --cache_latents \
    --source_column="source" \
    --target_column="target" \
    --caption_column="text" \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="bf16" \
    --train_data_dir=$TRAIN_DATA \
    --learning_rate=5e-5 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --num_validation_images=2 \
    --validation_image $VALIDATION_IMAGE \
    --num_train_epochs=50 \
    --validation_steps=200 \
    --checkpointing_steps=1000 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=500 \
    --max_grad_norm=1.0 \
    --report_to="wandb" \
    --seed=42