dataset="paws"
split="train"
include_deleted_spans=true

if [ "$include_deleted_spans" = true ]; then
  del_span_suffix="include_del_spans"
  deleted_spans_flag="--include_deleted_spans"
else
  del_span_suffix="no_del_spans"
  deleted_spans_flag=""  # Don't include the flag if false
fi

output_file="input/$dataset/${split}_with_graph_${del_span_suffix}.json"
label_map_file="input/label_map.json"
tokenizer_name="bert-base-uncased"
amr_cache_file="cache_files/paws_AMR_${split}.jsonl"

python preprocess_main.py \
  --dataset $dataset \
  --split $split \
  --output_file $output_file \
  --label_map_file $label_map_file \
  --tokenizer_name $tokenizer_name \
  --use_open_vocab \
  --max_seq_length 128 \
  --use_pointing \
  --with_graph \
  --amr_cache_file $amr_cache_file \
  $deleted_spans_flag