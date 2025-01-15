import argparse
import csv
import logging

from transformers import BertConfig

import predict
import utils


def batch_generator(
    dataset_dir,
    split,
    predict_batch_size,
):
    """Produces batches for predictions."""
    source_batch = []
    target_batch = []
    for source, target in utils.yield_sources_and_targets(
        dataset_dir,
        split,
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
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--predict_output_file", type=str, required=True)
    parser.add_argument("--label_map_file", type=str, required=True)
    parser.add_argument("--vocab_file", type=str, required=True)
    parser.add_argument("--bert_config_tagging", type=str, required=True)
    parser.add_argument("--bert_config_insertion", type=str, required=True)
    parser.add_argument("--model_tagging_filepath", type=str, required=True)
    parser.add_argument("--model_insertion_filepath", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
    )
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

    bert_config_tagging = BertConfig.from_json_file(args.bert_config_tagging)
    bert_config_insertion = BertConfig.from_json_file(
        args.bert_config_insertion
    )

    predictor = predict.FelixPredictor(
        bert_config_tagging=bert_config_tagging,
        bert_config_insertion=bert_config_insertion,
        model_tagging_filepath=args.model_tagging_filepath,
        model_insertion_filepath=args.model_insertion_filepath,
        vocab_file=args.vocab_file,
        label_map_file=args.label_map_file,
        sequence_length=args.max_seq_length,
        do_lowercase=args.do_lower_case,
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
            args.dataset_dir, args.split, args.predict_batch_size
        ):
            (
                predicted_tags,
                predicted_inserts,
            ) = predictor.predict_end_to_end_batch(source_batch)
            print(predicted_tags)
            print(predicted_inserts)
            num_predicted += len(source_batch)
            logging.info(f"{num_predicted} predicted.")
            for (
                source_input,
                target_output,
                predicted_tag,
                predicted_insert,
            ) in zip(
                source_batch, target_batch, predicted_tags, predicted_inserts
            ):
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
