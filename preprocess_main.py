import argparse
import json
import os
import random

from datasets import load_dataset

import preprocess
import utils

_INSERTION_FILENAME_SUFFIX = ".ins"


def perturb_text(text, delete_prob=0.1, shuffle_sentences=True):
    """
    Introduce artificial edits to the text for pretraining.

    Args:
        text: str, original text
        delete_prob: probability to drop a token
        shuffle_sentences: whether to shuffle sentences
    Returns:
        perturbed_text: str
    """
    # Token-level deletion
    tokens = text.split()
    tokens = [t for t in tokens if random.random() > delete_prob]

    # Sentence-level shuffle
    if shuffle_sentences:
        # split by period, shuffle, join
        sentences = " ".join(tokens).split(". ")
        random.shuffle(sentences)
        perturbed_text = ". ".join(sentences)
    else:
        perturbed_text = " ".join(tokens)

    return perturbed_text


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


def load_amr_cache(amr_cache_file):
    """Loads the AMR cache from a file.

    Args:
        amr_cache_file: Path to the AMR cache file.

    Returns:
        A dictionary where the key is the source sentence and the value is the AMR information.
    """
    if amr_cache_file and os.path.exists(amr_cache_file):
        amr_cache = {}
        print(f"Loading AMR cache from {amr_cache_file}")
        with open(amr_cache_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                source = entry["source"]
                amr_cache[source] = {
                    "amr_source": entry["amr_source"],
                    "amr_target": entry["amr_target"],
                }
        return amr_cache
    return None


def main(args):

    amr_cache = load_amr_cache(args.amr_cache_file)

    builder = preprocess.initialize_builder(
        args.use_open_vocab,
        args.label_map_file,
        args.max_seq_length,
        args.tokenizer_name,
        args.special_glue_string_for_joining_sources,
        args.with_graph,
        args.include_deleted_spans,
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
    elif "qqppos" in args.dataset:
        dataset = load_dataset(
            args.dataset, data_files={args.split: f"{args.split}.csv.gz"}
        )
        dataset = dataset[args.split]
    elif args.dataset == "c4":
        # Stream the dataset without loading fully into memory
        dataset = load_dataset(
            "c4", "en", split=args.split, trust_remote_code=True, streaming=True
        )
        if args.max_input_lines:
            dataset = dataset.take(args.max_input_lines)
    else:
        dataset = load_dataset(args.dataset, split=args.split)

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

    indexes = None
    if args.max_input_lines and args.dataset != "c4":
        input_len = sum(1 for _ in utils.yield_inputs(dataset))
        max_len = min(input_len, args.max_input_lines)
        indexes = set(random.sample(range(input_len), max_len))

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    insertion_path = args.output_file + _INSERTION_FILENAME_SUFFIX
    with open(args.output_file, "wb") as writer:
        with open(insertion_path, "wb") as writer_insertion:
            for i, inputs in enumerate(utils.yield_inputs(dataset)):
                (sources, target, amr_source, amr_target) = inputs
                if indexes and i not in indexes:
                    continue
                if target is None or not target.strip():
                    continue
                if i % 10000 == 0:
                    print(f"{i} examples processed, {num_converted} converted.")

                if args.dataset == "c4":
                    sources = perturb_text(
                        sources[0], delete_prob=0.2, shuffle_sentences=True
                    )
                    sources = [sources]
                example, insertion_examples = builder.build_transformer_example(
                    sources,
                    target,
                    amr_source,
                    amr_target,
                    use_token_type_ids=args.use_token_type_ids,
                    enhance_with_paraphrases=args.enhance_with_paraphrases,
                )
                if example is not None and insertion_examples is not None:
                    json_str = json.dumps(example.to_dict())
                    writer.write(json_str.encode("utf-8"))
                    writer.write(b"\n")
                    num_converted += 1

                    if isinstance(insertion_examples, dict):
                        insertion_examples = [insertion_examples]

                    if len(insertion_examples) == 0:
                        breakpoint()

                    for insertion_example in insertion_examples:
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
        "--insert_after_token", type=str, help="Token after which to insert."
    )
    parser.add_argument(
        "--with_graph",
        action="store_true",
        help="Whether to use graph information or not.",
    )
    parser.add_argument(
        "--amr_cache_file",
        type=str,
        default=None,
        help="The file path to the AMR cache file.",
    )
    parser.add_argument(
        "--include_deleted_spans",
        action="store_true",
        help="Whether to include deleted spans in processing.",
    )
    parser.add_argument(
        "--use_token_type_ids",
        action="store_true",
        help="Whether to use token_type_ids in the dataset",
    )
    parser.add_argument(
        "--enhance_with_paraphrases",
        action="store_true",
        help="Whether to enhance examples with WordNet paraphrases for unmapped phrases.",
    )

    args = parser.parse_args()

    if not args.use_open_vocab:
        raise ValueError("Currently only use_open_vocab=True is supported")

    main(args)
