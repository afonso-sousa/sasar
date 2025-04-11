import json
import os

import amrlib
from datasets import load_dataset

CACHE_DIR = "cache_files"
CACHE_FILE_TEMPLATE = "amr_cache_{}.jsonl"


def extract_amr(sentences, model_dir="amr_parser", device="cuda:0", batch_size=32):
    """Extract AMR graphs for a list of sentences using amrlib."""
    stog = amrlib.load_stog_model(
        model_dir=model_dir, device=device, batch_size=batch_size
    )
    graphs = stog.parse_sents(sentences)
    return graphs


def cache_amr_graphs(dataset_name, split="train", output_file=None, batch_size=32):
    """Loads a dataset, extracts AMR graphs, and caches them in a file."""
    if dataset_name == "paws":
        dataset = load_dataset(dataset_name, "labeled_final", split=split)
        dataset = dataset.filter(lambda x: x["label"] == 1)
        dataset = dataset.rename_column("sentence1", "source")
        dataset = dataset.rename_column("sentence2", "target")
    elif "qqppos" in args.dataset:
        dataset = load_dataset(
            args.dataset, data_files={args.split: f"{args.split}.csv.gz"}
        )
        dataset = dataset[args.split]
    else:
        dataset = load_dataset(dataset_name, split=split)

    # dataset = dataset.select(range(10))

    output_file = output_file or CACHE_FILE_TEMPLATE.format(split)
    sources = dataset["source"]
    targets = dataset["target"]

    print("Extracting AMR graphs for source sentences...")
    amr_sources = extract_amr(sources, batch_size=batch_size)

    print("Extracting AMR graphs for target sentences...")
    amr_targets = extract_amr(targets, batch_size=batch_size)

    with open(output_file, "w") as f:
        for src, amr_src, tgt, amr_tgt in zip(
            sources, amr_sources, targets, amr_targets
        ):
            json.dump(
                {
                    "source": src,
                    "amr_source": amr_src,
                    "target": tgt,
                    "amr_target": amr_tgt,
                },
                f,
            )
            f.write("\n")

    print(f"Saved AMR cache to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cache AMR graphs for a dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split.")
    parser.add_argument("--output_file", type=str, help="Output cache file.")
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for AMR parsing."
    )

    args = parser.parse_args()
    os.makedirs(CACHE_DIR, exist_ok=True)
    output_file = os.path.join(CACHE_DIR, args.output_file)
    cache_amr_graphs(args.dataset, args.split, output_file, args.batch_size)
