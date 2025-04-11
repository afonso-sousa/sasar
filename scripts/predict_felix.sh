dataset="raw-data/qqppos" # "paws"
split="test"
output_dir="output"
model_name="bert-base-uncased" # "answerdotai/ModernBERT-base" # "bert-base-uncased"
arch_name="felix"

main_dir=$output_dir/$(basename "$dataset")/$model_name/$arch_name

CUDA_VISIBLE_DEVICES=0 python predict_main.py \
  --dataset $dataset \
  --split $split \
  --tokenizer_name $model_name \
  --predict_output_file $main_dir/pred.tsv \
  --label_map_file input/label_map.json \
  --max_seq_length 256 \
  --predict_batch_size 32 \
  --use_open_vocab \
  --model_path $main_dir \
  --use_pointing \
  --use_token_type_ids