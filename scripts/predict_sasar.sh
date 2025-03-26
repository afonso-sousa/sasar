dataset="paws"
split="test"
output_dir="output"
model_name="answerdotai/ModernBERT-base" # "bert-base-uncased"

include_deleted_spans=true
if [ "$include_deleted_spans" = true ]; then
  del_span_suffix="include_del_spans"
  deleted_spans_flag=""
else
  del_span_suffix="no_del_spans"
  deleted_spans_flag="--no_deleted_spans"
fi
arch_name="sasar_${del_span_suffix}" # sasar_no_del_spans

main_dir=$output_dir/$model_name/$arch_name

CUDA_VISIBLE_DEVICES=0 python predict_main.py \
  --dataset $dataset \
  --split $split \
  --predict_output_file $main_dir/pred.tsv \
  --label_map_file input/label_map.json \
  --max_seq_length 256 \
  --predict_batch_size 32 \
  --use_open_vocab \
  --model_tagging_filepath $main_dir/tagger \
  --model_insertion_filepath $main_dir/inserter \
  --use_pointing \
  --with_graph \
  $deleted_spans_flag