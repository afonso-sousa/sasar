import argparse
import csv
import json
import logging
import os

from datasets import load_dataset

import predict
import preprocess
import utils


def batch_generator(
    dataset,
    predict_batch_size,
):
    """Produces batches for predictions."""
    source_batch = []
    target_batch = []
    for source, target, _, _ in utils.yield_inputs(
        dataset,
    ):
        source_batch.append(source[0])
        target_batch.append(target)
        if len(source_batch) == predict_batch_size:
            yield source_batch, target_batch
            source_batch = []
            target_batch = []

    # last batch (smaller then predict_batch_size)
    if source_batch:
        yield source_batch, target_batch


def get_predictor(model_path, tokenizer_name, label_map_file, args):
    """Initialize the appropriate predictor based on model directory structure."""

    tagger_path = os.path.join(model_path, "tagger")
    inserter_path = os.path.join(model_path, "inserter")

    if os.path.exists(tagger_path) and os.path.exists(inserter_path):
        # Split model case
        logging.info("Loading split model (tagger + inserter)")
        return predict.Predictor(
            model_tagging_filepath=tagger_path,
            model_insertion_filepath=inserter_path,
            tokenizer_name=tokenizer_name,
            label_map_file=label_map_file,
            sequence_length=args.max_seq_length,
            use_open_vocab=args.use_open_vocab,
            is_pointing=args.use_pointing,
            special_glue_string_for_joining_sources=args.special_glue_string_for_joining_sources,
            use_token_type_ids=args.use_token_type_ids,
            no_deleted_spans=args.no_deleted_spans,
        )
    else:
        # Joint model case
        logging.info("Loading joint model")
        return predict.JointPredictor(
            model_filepath=model_path,
            tokenizer_name=tokenizer_name,
            label_map_file=label_map_file,
            sequence_length=args.max_seq_length,
            use_open_vocab=args.use_open_vocab,
            is_pointing=args.use_pointing,
            special_glue_string_for_joining_sources=args.special_glue_string_for_joining_sources,
            use_token_type_ids=args.use_token_type_ids,
            no_deleted_spans=args.no_deleted_spans,
        )


def filter_dataset_by_masks(
    dataset,
    use_open_vocab,
    label_map_file,
    sequence_length,
    tokenizer_name,
    special_glue_string_for_joining_sources,
    no_deleted_spans,
    use_token_type_ids,
):
    """Filter dataset using the predictor's build_transformer_example method. Only working for joint model for now."""
    filtered_indices = []

    builder = preprocess.initialize_builder(
        use_open_vocab=use_open_vocab,
        label_map_file=label_map_file,
        max_seq_length=sequence_length,
        tokenizer_name=tokenizer_name,
        special_glue_string_for_sources=special_glue_string_for_joining_sources,
        with_graph=True,
        include_deleted_spans=not no_deleted_spans,
    )

    # TODO: This is temporary code for testing some scenarios. It should be replaced with a proper cache loading mechanism.
    amr_cache = {}
    with open("cache_files/paws_AMR_test.jsonl", "r") as f:
        for line in f:
            entry = json.loads(line)
            source = entry["source"]
            amr_cache[source] = {
                "amr_source": entry["amr_source"],
                "amr_target": entry["amr_target"],
            }

    if amr_cache:

        def add_amr_info(example):
            source = example["source"]
            if source in amr_cache:
                example["amr_source"] = amr_cache[source]["amr_source"].split("\n", 1)[
                    1
                ]
                example["amr_target"] = amr_cache[source]["amr_target"].split("\n", 1)[
                    1
                ]
            else:
                example["amr_source"] = ""
                example["amr_target"] = ""
            return example

        dataset = dataset.map(add_amr_info)

    for i, item in enumerate(dataset):
        source = item["source"]
        target = item["target"]
        amr_source = item["amr_source"]
        amr_target = item["amr_target"]

        example, insertion_example = builder.build_transformer_example(
            [source],
            target=target,
            amr_source=amr_source,
            amr_target=amr_target,
            is_test_time=False,
            use_token_type_ids=use_token_type_ids,
        )
        # If example is None, it means it exceeded the max masks allowed by label_map
        if example is not None and insertion_example is not None:
            filtered_indices.append(i)

    return dataset.select(filtered_indices)


def main():
    parser = argparse.ArgumentParser()
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split containing examples to be converted.",
    )
    parser.add_argument("--predict_output_file", type=str, required=True)
    parser.add_argument("--label_map_file", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model directory. For split models, should contain 'tagger' and 'inserter' subdirectories.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        required=True,
        help="Name of tokenizer to use.",
    )
    # Model configuration arguments
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument(
        "--use_open_vocab",
        action="store_true",
    )
    parser.add_argument(
        "--use_pointing",
        action="store_true",
    )
    parser.add_argument(
        "--special_glue_string_for_joining_sources", type=str, default="[SEP]"
    )
    parser.add_argument(
        "--use_token_type_ids",
        action="store_true",
        help="Whether to use token_type_ids in the dataset",
    )
    parser.add_argument(
        "--no_deleted_spans",
        action="store_true",
        help="Whether to not include deleted spans in processing.",
    )
    # Runtime arguments
    parser.add_argument(
        "--predict_batch_size",
        default=32,
        type=int,
        help="Batch size for the prediction of insertion and tagging models.",
    )
    parser.add_argument(
        "--limit_masks",
        action="store_true",
        help="Whether to filter out examples with more consecutive masks than in the label map.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not args.use_open_vocab:
        raise ValueError("Currently only use_open_vocab=True is supported")

    if args.dataset == "paws":
        dataset = load_dataset(args.dataset, "labeled_final", split=args.split)
        dataset = dataset.filter(lambda x: x["label"] == 1)
        dataset = dataset.rename_column("sentence1", "source")
        dataset = dataset.rename_column("sentence2", "target")
    elif "qqppos" in args.dataset:
        dataset = load_dataset(
            args.dataset, data_files={args.split: f"{args.split}.csv.gz"}
        )
        dataset = dataset[args.split]
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    # path_components = args.model_path.split(os.sep)
    # tokenizer_name = os.path.join(*path_components[1:3])

    predictor = get_predictor(
        args.model_path, args.tokenizer_name, args.label_map_file, args
    )

    # Filter dataset if limit_masks is specified
    if args.limit_masks is not None:
        dataset = filter_dataset_by_masks(
            dataset,
            args.use_open_vocab,
            args.label_map_file,
            args.max_seq_length,
            args.tokenizer_name,
            args.special_glue_string_for_joining_sources,
            args.no_deleted_spans,
            args.use_token_type_ids,
        )

        logging.info(f"Filtered dataset size: {len(dataset)}")

    num_predicted = 0

    with open(args.predict_output_file, "w") as writer:
        tsv_writer = csv.writer(writer, delimiter="\t")
        header = ["source", "predicted_tags", "predicted_insertions", "target"]
        tsv_writer.writerow(header)
        for source_batch, target_batch in batch_generator(
            dataset, args.predict_batch_size
        ):
            (
                predicted_tags,
                predicted_inserts,
            ) = predictor.predict_end_to_end_batch(source_batch)
            num_predicted += len(source_batch)
            logging.info(f"{num_predicted} predicted.")
            for (
                source_input,
                target_output,
                predicted_tag,
                predicted_insert,
            ) in zip(source_batch, target_batch, predicted_tags, predicted_inserts):
                tsv_writer.writerow(
                    [
                        source_input,
                        predicted_tag,
                        predicted_insert,
                        target_output,
                    ]
                )


if __name__ == "__main__":
    main()
