# SASAR

SASAR is flexible text-editing approach for generation, designed to derive
maximum benefit from the ideas of decoding with bi-directional contexts and
self-supervised pretraining. We achieve this by decomposing the text-editing
task into two sub-tasks: **tagging** to decide on the subset of input tokens and
their order in the output text and **insertion** to in-fill the missing tokens in
the output not present in the input.

## Usage Instructions

Running an experiment with SASAR consists of the following steps:

1. Create label_map for tagging model
2. Convert data for insertion/tagging model.
3. Finetune the tagging/insertion models.
4. Compute predictions.


### 1. Label map construction


```
# Label map construction

sh ./scripts/vocabulary_constructor.sh
```

### 2. Converting data for insertion/tagging model

The PAWS dataset can be found in the [official repository](https://github.com/google-research-datasets/paws) or for ease of use through the [HuggingFace Hub](https://huggingface.co/datasets/paws)

```
# Preprocess

python preprocess_main.py \
  --dataset_dir "paws" \
  --split "train" \
  --output_file "input/train.json" \
  --label_map_file "input/label_map.json" \
  --vocab_file="input/vocab.txt" \
  --do_lower_case \
  --use_open_vocab \
  --max_seq_length 128 \
  --use_pointing

python preprocess_main.py \
  --dataset_dir "paws" \
  --split "validation" \
  --output_file "input/validation.json" \
  --label_map_file "input/label_map.json" \
  --vocab_file="input/vocab.txt" \
  --do_lower_case \
  --use_open_vocab \
  --max_seq_length 128 \
  --use_pointing
```

### 3. Model Training

Model hyperparameters are specified in [text_editing_config.json](input/text_editing_config.json).
**note** These models can be trained independently, as such it is quicker to train them in parallel rather than sequentially.


Train the models on CPU/GPU.

```
# Train tagger
python run.py \
    --output_dir output/tagger \
    --train_file input/train.json \
    --validation_file input/validation.json \
    --bert_config_tagging input/text_editing_config.json \
    --vocab_file input/vocab.txt \
    --label_map_file input/label_map.json \
    --max_seq_length 128 \
    --num_train_epochs 500 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --use_pointing \
    --learning_rate 3e-5 \
    --pointing_weight 1 \
    --use_weighted_labels \
    --use_open_vocab

# Train inserter
python run.py \
    --output_dir output/inserter \
    --train_file input/train.json.ins \
    --validation_file input/validation.json.ins \
    --bert_config_insertion input/text_editing_config.json \
    --vocab_file input/vocab.txt \
    --label_map_file input/label_map.json \
    --max_seq_length 128 \
    --num_train_epochs 500 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --use_pointing \
    --learning_rate 3e-5 \
    --pointing_weight 1 \
    --use_open_vocab \
    --train_insertion
```

### 4. Prediction


```
# Predict

python predict_main.py \
  --dataset_dir "paws" \
  --split "test" \
  --predict_output_file output/pred.tsv \
  --label_map_file input/label_map.json \
  --vocab_file input/vocab.txt \
  --max_seq_length 128 \
  --predict_batch_size 32 \
  --do_lower_case \
  --use_open_vocab \
  --bert_config_tagging input/text_editing_config.json \
  --bert_config_insertion input/text_editing_config.json \
  --model_tagging_filepath output/tagger \
  --model_insertion_filepath output/inserter \
  --use_pointing
```

The predictions output a TSV file with four columns: Source, the input to the insertion model, the final output, and the reference. Note the felix output is tokenized (WordPieces), including a start "[CLS]" and end "[SEP]". WordPieces can be removed by replacing " ##" with "". Additionally words have been split on punctuation "don't -> don ' t", this must also be reversed.

## Running Unit Tests

You can run all unit tests executing the following command:
```
python -m unittest discover -p "*_test.py"
```


## License

MIT; see [LICENSE](LICENSE) for details.
