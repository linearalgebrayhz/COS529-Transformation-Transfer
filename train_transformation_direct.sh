#!/bin/bash

# Base configuration for transformation learning training
export PRETRAINED_MODEL="black-forest-labs/FLUX.1-dev"
export FIRST_STAGE_MODEL="outputs/viewsynthesis_reduced_train"  # Path to your first stage model
export DATA_DIR="data/BlendedMVS/transformation_dataset.jsonl"
export VALIDATION_DIR="data/transformation/validation"
export CONFIG="/home/linux/.cache/huggingface/accelerate/default_config.yaml"  # Change to your config path

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# Clear any existing CUDA cache
python -c "import torch; torch.cuda.empty_cache()"

# Method 1: Direct Method Training
echo "Starting training with Direct method..."

export OUTPUT_DIR="outputs/transformation_direct"
export LOG_PATH="$OUTPUT_DIR/logs"

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_PATH

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file $CONFIG train_transformation_learning.py \
    --pretrained_model_name_or_path $PRETRAINED_MODEL \
    --first_stage_model_path $FIRST_STAGE_MODEL \
    --train_data_dir $DATA_DIR \
    --width 768 \
    --height 576 \
    --dataloader_num_workers 4 \
    --cache_latents \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="fp16" \
    --method="direct" \
    --validation_source_image $VALIDATION_DIR/triplet_0_test_source.jpg \
    --validation_reference_source $VALIDATION_DIR/triplet_0_source.jpg \
    --validation_reference_target $VALIDATION_DIR/triplet_0_target.jpg \
    --validation_transform "$VALIDATION_DIR/triplet_0_transform.json" \
    --rank=32 \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --num_validation_images=2 \
    --num_train_epochs=20 \
    --validation_steps=200 \
    --checkpointing_steps=500 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=200 \
    --seed=42