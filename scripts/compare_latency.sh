dataset_path="paws" # "raw-data/qqppos" # paws
sasar_model_path="output/bert-base-uncased/joint_sasar_include_del_spans"
metric="metrics/my_metric"

CUDA_VISIBLE_DEVICES=0 python compare_latency.py \
  --dataset_name $dataset_path \
  --sasar_model_path path/to/sasar/model \
  --sasar_tokenizer tokenizer_name \
  --sasar_label_map label_map_file \
  --t5_model_path path/to/t5/model \
  --batch_size 32