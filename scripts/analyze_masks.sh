input_file="input/paws/train_with_graph_no_del_spans_answerdotai/ModernBERT-base.json.ins"
tokenizer="answerdotai/ModernBERT-base"

python analyze_masks.py \
    --input_file $input_file \
    --tokenizer $tokenizer