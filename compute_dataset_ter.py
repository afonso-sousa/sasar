from typing import Dict

import evaluate
import numpy as np
from datasets import Dataset, load_dataset
from nltk.tokenize import word_tokenize


def load_and_prepare_dataset(dataset_name: str, split: str = "train") -> Dataset:
    """
    Load and prepare the dataset based on the given name.
    """
    if dataset_name == "paws":
        dataset = load_dataset(dataset_name, "labeled_final", split=split)
        dataset = dataset.filter(lambda x: x["label"] == 1)
        dataset = dataset.rename_column("sentence1", "source")
        dataset = dataset.rename_column("sentence2", "target")
    elif "qqppos" in dataset_name:
        dataset = load_dataset(dataset_name, data_files={split: f"{split}.csv.gz"})
        dataset = dataset[split]
    else:
        dataset = load_dataset(dataset_name, split=split)

    # Ensure the dataset has source and target columns
    if "source" not in dataset.column_names or "target" not in dataset.column_names:
        raise ValueError("Dataset must contain 'source' and 'target' columns")

    return dataset


def compute_stats(dataset) -> Dict[str, float]:
    """
    Compute TER statistics between source and target texts.

    Args:
        dataset: Dataset containing 'source' and 'target' columns

    Returns:
        Dictionary containing TER statistics
    """
    ter = evaluate.load("ter")
    sources = dataset["source"]
    targets = dataset["target"]

    # Compute average token lengths
    def avg_token_length(texts):
        lengths = [len(word_tokenize(text)) for text in texts]
        return np.mean(lengths)

    avg_source_len = avg_token_length(sources)
    avg_target_len = avg_token_length(targets)

    results = ter.compute(predictions=targets, references=sources)
    return {
        "average_ter": results["score"],
        "average_source_length": avg_source_len,
        "average_target_length": avg_target_len,
        "size": len(sources),
        "average_num_edits": results["num_edits"] / len(sources),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset to load"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to use"
    )
    args = parser.parse_args()

    try:
        # Load and prepare dataset
        dataset = load_and_prepare_dataset(args.dataset_name, args.split)
        print(f"Loaded dataset with {len(dataset)} examples")

        # Compute TER statistics
        stats = compute_stats(dataset)

        # Print results
        print("\nStatistics:")
        print(f"Average TER score: {stats['average_ter']:.4f}")
        print(f"Average source length: {stats['average_source_length']:.2f} tokens")
        print(f"Average target length: {stats['average_target_length']:.2f} tokens")
        print(f"Dataset size: {stats['size']} examples")
        print(f"Average number of edits: {stats['average_num_edits']:.2f} edits")

    except Exception as e:
        print(f"Error: {str(e)}")
