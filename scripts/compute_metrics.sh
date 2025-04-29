# file_path="output/paws/bert-base-uncased/felix/pred.tsv"
# file_path="output/paws/bert-base-uncased/sasar_include_del_spans/pred.tsv"
# file_path="output/paws/bert-base-uncased/sasar_no_del_spans/pred.tsv"
# file_path="output/paws/bert-base-uncased/joint_sasar_no_del_spans/pred.tsv"
# file_path="output/paws/bert-base-uncased/joint_sasar_include_del_spans/pred.tsv"
# file_path="output/paws/bert-base-uncased/1e-4/sasar_no_del_spans/pred.tsv"
file_path="output/paws/answerdotai/ModernBERT-base/1e-4/sasar_no_del_spans/pred.tsv"

# file_path="output/qqppos/bert-base-uncased/felix/pred.tsv"
# file_path="output/qqppos/bert-base-uncased/sasar_include_del_spans/pred.tsv"
# file_path="output/qqppos/bert-base-uncased/sasar_no_del_spans/pred.tsv"
# file_path="output/qqppos/bert-base-uncased/joint_sasar_no_del_spans/pred.tsv"
# file_path="output/qqppos/bert-base-uncased/joint_sasar_include_del_spans/pred.tsv"

metric="metrics/my_metric"

CUDA_VISIBLE_DEVICES=0 python compute_metrics.py \
  --file_path $file_path \
  --metric $metric