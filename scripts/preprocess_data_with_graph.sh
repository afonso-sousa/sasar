dataset="paws"
split="train"
output_file="input/$dataset/${split}.json"
label_map_file="input/label_map.json"
tokenizer_name="bert-base-uncased"

python preprocess_main.py \
  --dataset $dataset \
  --split $split \
  --output_file $output_file \
  --label_map_file $label_map_file \
  --tokenizer_name $tokenizer_name \
  --use_open_vocab \
  --max_seq_length 128 \
  --use_pointing \
  --with_graph