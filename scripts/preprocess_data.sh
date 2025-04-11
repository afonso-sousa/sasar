dataset="raw-data/qqppos" # paws
split="validation"
tokenizer_name="bert-base-uncased" # "answerdotai/ModernBERT-base" # "bert-base-uncased"
output_file="input/$(basename "$dataset")/${split}_${tokenizer_name}.json"
label_map_file="input/label_map.json"

python preprocess_main.py \
  --dataset $dataset \
  --split $split \
  --output_file $output_file \
  --label_map_file $label_map_file \
  --tokenizer_name $tokenizer_name \
  --use_open_vocab \
  --max_seq_length 128 \
  --use_pointing \
  --use_token_type_ids