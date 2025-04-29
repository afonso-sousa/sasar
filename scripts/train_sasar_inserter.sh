dataset_name="paws" # "paws" # "qqppos"
model_name="answerdotai/ModernBERT-base" # "answerdotai/ModernBERT-base" # "bert-base-uncased"
lr=1e-4

include_deleted_spans=false

if [ "$include_deleted_spans" = true ]; then
  del_span_suffix="include_del_spans"
else
  del_span_suffix="no_del_spans"
fi

arch_name="sasar_${del_span_suffix}" # sasar_no_del_spans

CUDA_VISIBLE_DEVICES=1 python train_sasar.py \
    --output_dir output/$dataset_name/$model_name/$lr/$arch_name/inserter \
    --train_file input/$dataset_name/train_with_graph_${del_span_suffix}_${model_name}.json.ins \
    --validation_file input/$dataset_name/validation_with_graph_${del_span_suffix}_${model_name}.json.ins \
    --model_name_or_path $model_name \
    --label_map_file input/label_map.json \
    --max_seq_length 256 \
    --num_train_epochs 500 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --use_pointing \
    --learning_rate $lr \
    --lr_scheduler_type linear \
    --num_warmup_steps 500 \
    --pointing_weight 1 \
    --model_type inserter \
    --patience 10