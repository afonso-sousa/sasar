dataset_name="paws" # "paws" # "qqppos"

model_name="answerdotai/ModernBERT-base" # "answerdotai/ModernBERT-base" # "bert-base-uncased"

include_deleted_spans=false

if [ "$include_deleted_spans" = true ]; then
  del_span_suffix="include_del_spans"
else
  del_span_suffix="no_del_spans"
fi

arch_name="joint_sasar_${del_span_suffix}"

CUDA_VISIBLE_DEVICES=0 python train_sasar.py \
    --output_dir output/$dataset_name/$model_name/$arch_name \
    --train_file input/$dataset_name/train_with_graph_${del_span_suffix}_${model_name}_joint.jsonl \
    --validation_file input/$dataset_name/validation_with_graph_${del_span_suffix}_${model_name}_joint.jsonl \
    --model_name_or_path $model_name \
    --label_map_file input/label_map.json \
    --max_seq_length 256 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --use_pointing \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --num_warmup_steps 500 \
    --pointing_weight 1 \
    --use_weighted_labels \
    --model_type joint \
    --patience 10 \
    --use_token_type_ids