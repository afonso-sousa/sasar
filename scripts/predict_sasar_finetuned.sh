dataset="paws" # "raw-data/qqppos" # "paws"
split="test"
tokenizer_name="answerdotai/ModernBERT-base" # "answerdotai/ModernBERT-base" # "bert-base-uncased"

arch_name="joint_sasar_${del_span_suffix}" # sasar_no_del_spans

main_dir="output/c4/answerdotai/ModernBERT-base/pretrained_sasar"

CUDA_VISIBLE_DEVICES=0 python predict_main.py \
  --dataset $dataset \
  --split $split \
  --tokenizer_name $tokenizer_name \
  --predict_output_file $main_dir/pred.tsv \
  --label_map_file input/label_map.json \
  --max_seq_length 256 \
  --predict_batch_size 32 \
  --use_open_vocab \
  --model_path $main_dir \
  --use_pointing \
  --no_deleted_spans