import argparse
import csv
import logging
import os

from datasets import load_dataset

import predict
import utils


def batch_generator(
    dataset,
    predict_batch_size,
):
    """Produces batches for predictions."""
    source_batch = []
    target_batch = []
    for source, target in utils.yield_sources_and_targets(
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


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--model_tagging_filepath", type=str, required=True)
    parser.add_argument("--model_insertion_filepath", type=str, required=True)
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
        "--predict_batch_size",
        default=32,
        type=int,
        help="Batch size for the prediction of insertion and tagging models.",
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
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    model_arch_name = os.path.basename(os.path.dirname(args.model_tagging_filepath))

    predictor = predict.FelixPredictor(
        model_tagging_filepath=args.model_tagging_filepath,
        model_insertion_filepath=args.model_insertion_filepath,
        tokenizer_name=model_arch_name,
        label_map_file=args.label_map_file,
        sequence_length=args.max_seq_length,
        use_open_vocab=args.use_open_vocab,
        is_pointing=args.use_pointing,
        special_glue_string_for_joining_sources=args.special_glue_string_for_joining_sources,
    )

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
