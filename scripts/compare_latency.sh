dataset_path="paws" # "raw-data/qqppos" # paws

sasar_model_path="output/$(basename "$dataset_path")/bert-base-uncased/joint_sasar_no_del_spans"
sasar_tokenizer="bert-base-uncased"

t5_model_path="output/$(basename "$dataset_path")/google/flan-t5-small"
t5_tokenizer="google/flan-t5-small"

metric="metrics/my_metric"

CUDA_VISIBLE_DEVICES=0 python compare_latency.py \
  --dataset_name $dataset_path \
  --sasar_model_path $sasar_model_path \
  --sasar_tokenizer $sasar_tokenizer \
  --sasar_label_map input/label_map.json \
  --t5_model_path $t5_model_path \
  --t5_tokenizer $t5_tokenizer \
  --batch_size 32