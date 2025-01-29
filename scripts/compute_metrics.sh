file_path="output/bert-base-uncased/pred.tsv"
metric="bleu"

CUDA_VISIBLE_DEVICES=0 python compute_metrics.py \
  --file_path $file_path \
  --metric $metric