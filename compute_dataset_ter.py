from typing import Dict, Optional

import evaluate
import numpy as np
from datasets import load_dataset


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


def compute_ter_stats(dataset, compute_details: bool = False) -> Dict[str, float]:
    """
    Compute TER statistics between source and target texts.

    Args:
        dataset: Dataset containing 'source' and 'target' columns
        compute_details: Whether to compute detailed operation statistics

    Returns:
        Dictionary containing TER statistics
    """
    ter = evaluate.load("ter")
    sources = dataset["source"]
    targets = dataset["target"]

    # Basic TER computation
    if not compute_details:
        scores = ter.compute(predictions=targets, references=sources)
        return {"average_ter": np.mean(scores["score"])}

    # Detailed TER computation with operation counts
    detailed_scores = []
    operations = {"insertions": [], "deletions": [], "substitutions": [], "shifts": []}

    for src, tgt in zip(sources, targets):
        result = ter.compute(predictions=[tgt], references=[src], detailed=True)
        detailed_scores.append(result["score"])

        if "ops" in result:
            ops = result["ops"]
            operations["insertions"].append(ops.get("insertions", 0))
            operations["deletions"].append(ops.get("deletions", 0))
            operations["substitutions"].append(ops.get("substitutions", 0))
            operations["shifts"].append(ops.get("shifts", 0))

    stats = {
        "average_ter": np.mean(detailed_scores),
        "average_insertions": np.mean(operations["insertions"]),
        "average_deletions": np.mean(operations["deletions"]),
        "average_substitutions": np.mean(operations["substitutions"]),
        "average_shifts": np.mean(operations["shifts"]),
    }

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset to load"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to use"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Compute detailed operation statistics"
    )
    args = parser.parse_args()

    try:
        # Load and prepare dataset
        dataset = load_and_prepare_dataset(args.dataset_name, args.split)
        print(f"Loaded dataset with {len(dataset)} examples")

        # Compute TER statistics
        stats = compute_ter_stats(dataset, args.detailed)

        # Print results
        print("\nTER Statistics:")
        print(f"Average TER score: {stats['average_ter']:.4f}")

        if args.detailed:
            print("\nDetailed Operation Averages:")
            print(f"Insertions: {stats['average_insertions']:.4f}")
            print(f"Deletions: {stats['average_deletions']:.4f}")
            print(f"Substitutions: {stats['average_substitutions']:.4f}")
            print(f"Shifts: {stats['average_shifts']:.4f}")

    except Exception as e:
        print(f"Error: {str(e)}")
