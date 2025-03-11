dataset="paws"
split="test"
output_dir="output/answerdotai/ModernBERT-base"

CUDA_VISIBLE_DEVICES=0 python predict_main.py \
  --dataset $dataset \
  --split $split \
  --predict_output_file $output_dir/pred.tsv \
  --label_map_file input/label_map.json \
  --max_seq_length 128 \
  --predict_batch_size 32 \
  --use_open_vocab \
  --model_tagging_filepath $output_dir/tagger_with_graph \
  --model_insertion_filepath $output_dir/inserter_with_graph \
  --use_pointing