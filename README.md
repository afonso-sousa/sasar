# SASAR

SASAR is flexible text-editing approach for generation, designed to derive
maximum benefit from the ideas of decoding with bi-directional contexts and
self-supervised pretraining. We achieve this by decomposing the text-editing
task into two sub-tasks: **tagging** to decide on the subset of input tokens and
their order in the output text and **insertion** to in-fill the missing tokens in
the output not present in the input.

## Setup environment

### Download pseudo-semantic graphs
We use the pseudo-semantic graphs from https://github.com/afonso-sousa/sem_para_gen.git. Get the folder /pseudo_semantic_graph into the project's root folder.

## Usage Instructions

Running an experiment with SASAR consists of the following steps:

1. Create label_map for tagging model
2. Convert data for insertion/tagging model.
3. Finetune the tagging/insertion models.
4. Compute predictions.

*But, don't worry! Everything is readily available as bash scripts.*


### 1. Label map construction
```
sh ./scripts/vocabulary_constructor.sh
```

### 2. Converting data for insertion/tagging model

The PAWS dataset can be found in the [official repository](https://github.com/google-research-datasets/paws) or for ease of use through the [HuggingFace Hub](https://huggingface.co/datasets/paws)

```
sh scripts/preprocess_data.sh
```

You may run it twice for train and validation sets.

### 3. Model Training

Run the following command to train the tagger:
```
sh ./script/train_tagger.sh
```

Or this command to train the inserter:
```
sh ./script/train_inserter.sh
```

**Note** These models can be trained independently, as such it is quicker to train them in parallel rather than sequentially.

### 4. Prediction
```
sh ./scripts/predict.sh
```

The predictions output a TSV file with four columns: Source, the input to the insertion model, the final output, and the reference. Note the felix output is tokenized (WordPieces), including a start "[CLS]" and end "[SEP]". WordPieces can be removed by replacing " ##" with "". Additionally words have been split on punctuation "don't -> don ' t", this must also be reversed.

## Running Unit Tests

You can run all unit tests executing the following command:
```
python -m unittest discover -p "*_test.py"
```

## License

MIT; see [LICENSE](LICENSE) for details.
