dataset="paws"
split="test"
output_dir="output"
model_name="answerdotai/ModernBERT-base" # "bert-base-uncased"
arch_name="felix"

CUDA_VISIBLE_DEVICES=0 python predict_main.py \
  --dataset $dataset \
  --split $split \
  --predict_output_file $output_dir/$model_name/$arch_name/pred.tsv \
  --label_map_file input/label_map.json \
  --max_seq_length 256 \
  --predict_batch_size 32 \
  --use_open_vocab \
  --model_tagging_filepath $output_dir/$model_name/$arch_name/tagger \
  --model_insertion_filepath $output_dir/$model_name/$arch_name/inserter \
  --use_pointing