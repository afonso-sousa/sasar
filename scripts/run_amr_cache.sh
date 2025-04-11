#!/bin/bash

DATASET="raw-data/qqppos"
SPLIT="train" # validation
OUTPUT_FILE="$(basename "$DATASET")_AMR_${SPLIT}.jsonl"
BATCH_SIZE=64

echo "Running AMR extraction on dataset: $DATASET (split: $SPLIT)"
echo "Caching to: $OUTPUT_FILE"

python cache_amrs.py --dataset $DATASET --split $SPLIT --output_file $OUTPUT_FILE --batch_size $BATCH_SIZE
