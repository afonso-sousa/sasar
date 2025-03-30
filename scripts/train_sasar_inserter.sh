model_name="answerdotai/ModernBERT-base" # "bert-base-uncased"

include_deleted_spans=true

if [ "$include_deleted_spans" = true ]; then
  del_span_suffix="include_del_spans"
else
  del_span_suffix="no_del_spans"
fi

arch_name="sasar_${del_span_suffix}" # sasar_no_del_spans

CUDA_VISIBLE_DEVICES=1 python train_sasar.py \
    --output_dir output/$model_name/$arch_name/inserter \
    --train_file input/paws/train_with_graph_${del_span_suffix}.json.ins \
    --validation_file input/paws/validation_with_graph_${del_span_suffix}.json.ins \
    --model_name_or_path $model_name \
    --label_map_file input/label_map.json \
    --max_seq_length 128 \
    --num_train_epochs 500 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --use_pointing \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --num_warmup_steps 500 \
    --pointing_weight 1 \
    --model_type inserter \
    --patience 10