file_path="output/answerdotai/ModernBERT-base/felix/pred.tsv"
metric="metrics/my_metric"

CUDA_VISIBLE_DEVICES=0 python compute_metrics.py \
  --file_path $file_path \
  --metric $metric