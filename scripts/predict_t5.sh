dataset_name="paws"
split="test"
model_name="google/flan-t5-small"
output_dir="output"
metric="metrics/my_metric"

CUDA_VISIBLE_DEVICES=1 python test_t5.py \
  --dataset_name $dataset_name \
  --model_name_or_path $output_dir/$model_name \
  --split $split \
  --predict_output_file $output_dir/$model_name/pred.tsv \
  --max_seq_length 128 \
  --metric $metric \
  --predict_batch_size 32