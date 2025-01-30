import argparse
import json
import os
import random

from datasets import load_dataset

import preprocess
import utils

_INSERTION_FILENAME_SUFFIX = ".ins"


def _write_example_count(count, example_path):
    """Saves the number of converted examples to a file.

    This count is used when determining the number of training steps.

    Args:
        count: The number of converted examples.
        example_path: Path to the file where the examples are saved.

    Returns:
        The path to which the example count is saved
        (example_path + '.num_examples.txt').
    """
    count_fname = example_path + ".num_examples.txt"
    with open(count_fname, "w") as count_writer:
        count_writer.write(str(count))
    return count_fname


def main(args):
    builder = preprocess.initialize_builder(
        args.use_open_vocab,
        args.label_map_file,
        args.max_seq_length,
        args.tokenizer_name,
        args.special_glue_string_for_joining_sources,
        args.with_graph,
    )

    num_converted = 0
    num_converted_insertion = 0
    random.seed(42)

    # load dataset
    if args.dataset == "paws":
        dataset = load_dataset(args.dataset, "labeled_final", split=args.split)
        dataset = dataset.filter(lambda x: x["label"] == 1)
        dataset = dataset.rename_column("sentence1", "source")
        dataset = dataset.rename_column("sentence2", "target")
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    indexes = None
    if args.max_input_lines:
        input_len = sum(1 for _ in utils.yield_sources_and_targets(dataset))
        max_len = min(input_len, args.max_input_lines)
        indexes = set(random.sample(range(input_len), max_len))

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    insertion_path = args.output_file + _INSERTION_FILENAME_SUFFIX
    with open(args.output_file, "wb") as writer:
        with open(insertion_path, "wb") as writer_insertion:
            for i, (sources, target) in enumerate(
                utils.yield_sources_and_targets(dataset)
            ):
                if indexes and i not in indexes:
                    continue
                if target is None or not target.strip():
                    continue
                if i % 10000 == 0:
                    print(f"{i} examples processed, {num_converted} converted.")

                example, insertion_example = builder.build_transformer_example(
                    sources, target
                )
                if example is not None:
                    json_str = json.dumps(example.to_dict())
                    writer.write(json_str.encode("utf-8"))
                    writer.write(b"\n")
                    num_converted += 1
                if insertion_example is not None:
                    json_str = json.dumps(insertion_example)
                    writer_insertion.write(json_str.encode("utf-8"))
                    writer_insertion.write(b"\n")
                    num_converted_insertion += 1

    print(
        f"Done. {num_converted} tagging and {num_converted_insertion} insertion examples."
    )
    count_fname = _write_example_count(num_converted, args.output_file)
    insertion_count_fname = _write_example_count(
        num_converted_insertion, insertion_path
    )
    print(
        "\n".join(
            [
                "Wrote:",
                args.output_file,
                count_fname,
                insertion_path,
                insertion_count_fname,
            ]
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a dataset into the TFRecord format."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split containing examples to be converted.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output file.",
    )
    parser.add_argument("--max_input_lines", type=int, help="Number of samples.")
    parser.add_argument(
        "--use_pointing",
        action="store_true",
        help="Whether to use pointing or not.",
    )
    parser.add_argument(
        "--use_open_vocab",
        action="store_true",
        help="Whether to use open vocabulary or not.",
    )
    parser.add_argument(
        "--label_map_file",
        type=str,
        required=True,
        help="Path to the label map file.",
    )
    parser.add_argument("--max_seq_length", type=int, help="Maximum sequence length.")
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        help="Name of HF tokenizer.",
    )
    parser.add_argument(
        "--special_glue_string_for_joining_sources",
        type=str,
        help="Special glue string for joining sources.",
    )
    parser.add_argument(
        "--max_mask",
        type=int,
        help="Maximum number of masked tokens in the sequence.",
    )
    parser.add_argument(
        "--insert_after_token", type=str, help="Token after which to insert."
    )
    parser.add_argument(
        "--with_graph",
        action="store_true",
        help="Whether to use graph information or not.",
    )

    args = parser.parse_args()

    if not args.use_open_vocab:
        raise ValueError("Currently only use_open_vocab=True is supported")

    main(args)
