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
This just creates a dictionary with the Tagger labels.

### 2. Converting data for insertion/tagging model

We use two datasets: the PAWS dataset that can be found in the [official repository](https://github.com/google-research-datasets/paws) or for ease of use through the [HuggingFace Hub](https://huggingface.co/datasets/paws); and the Quora Question Pairs that can be found in the [official post](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) or using the [HuggingFace Hub](https://huggingface.co/datasets/sentence-transformers/quora-duplicates) and the subset `pair`.

```
# For Felix-type data preprocessing.
sh ./scripts/preprocess_data.sh
```

Or

```
# For SASAR-type data preprocessing.
sh ./scripts/preprocess_data_with_data.sh
```

For the latter, you may want to pre-extract the AMR graphs (altough not required). You can do so running:
```
sh ./scripts/run_amr_cache.sh
```
Additionally, you will need to fuse the tagging and insertion datasets with:
```
python join_tag_and_insert_data.py
```

You may run it twice for train and validation sets.

### 3. Model Training

You have a lot a flavours of models to train. Just check [scripts](https://github.com/afonso-sousa/sasar/tree/main/scripts) and search for `train_sasar_*.sh`.

For split tagger and inserter, the models can be trained independently, so it might be quicker to train them in parallel rather than sequentially.

### 4. Prediction
Same goes for prediction. Check [scripts](https://github.com/afonso-sousa/sasar/tree/main/scripts) and search for `predict_sasar_*.sh`.

## License

MIT; see [LICENSE](LICENSE) for details.
