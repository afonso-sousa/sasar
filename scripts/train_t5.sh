#!/bin/bash

model_name="google/flan-t5-small" # "google/flan-t5-base"
dataset_name="paws"

# Training command
CUDA_VISIBLE_DEVICES=0 python train_t5.py \
    --output_dir output/$model_name \
    --dataset_name $dataset_name \
    --model_name_or_path $model_name \
    --max_seq_length 128 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --num_warmup_steps 500