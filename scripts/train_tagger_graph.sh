model_name=bert-base-uncased

CUDA_VISIBLE_DEVICES=0 python train_sasar.py \
    --output_dir output/$model_name/tagger \
    --train_file input/paws/train.json \
    --validation_file input/paws/validation.json \
    --model_name_or_path $model_name \
    --label_map_file input/label_map.json \
    --max_seq_length 128 \
    --num_train_epochs 50 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --use_pointing \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --num_warmup_steps 500 \
    --pointing_weight 1 \
    --use_weighted_labels \
    --use_open_vocab \
    --with_graph