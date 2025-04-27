dataset_path="raw-data/qqppos" # "raw-data/qqppos" # paws
split="train"
sasar_model_path="output/bert-base-uncased/joint_sasar_include_del_spans"
metric="metrics/my_metric"

python compute_dataset_ter.py \
  --dataset_name $dataset_path \
  --split $split