import argparse
import json
from collections import Counter
from statistics import mean

from transformers import AutoTokenizer


def analyze_mask_clusters(input_file, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id

    if mask_token_id is None:
        raise ValueError(
            f"The tokenizer '{tokenizer_name}' does not have a mask token."
        )

    print(f"Using tokenizer: {tokenizer_name}")
    print(f"MASK token ID: {mask_token_id}")
    print(f"PAD token ID: {pad_token_id}")

    num_entries = 0
    total_masks = 0
    max_consecutive = 0
    min_consecutive = float("inf")
    cluster_freq = Counter()
    total_length = 0

    with open(input_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            input_ids = entry.get("input_ids", [])
            total_length += len(input_ids)

            num_entries += 1
            current_cluster = 0
            masks_in_entry = 0

            for token in input_ids:
                if token == mask_token_id:
                    current_cluster += 1
                    masks_in_entry += 1
                elif current_cluster > 0:
                    cluster_freq[current_cluster] += 1
                    max_consecutive = max(max_consecutive, current_cluster)
                    min_consecutive = min(min_consecutive, current_cluster)
                    current_cluster = 0

            if current_cluster > 0:
                cluster_freq[current_cluster] += 1
                max_consecutive = max(max_consecutive, current_cluster)
                min_consecutive = min(min_consecutive, current_cluster)

            total_masks += masks_in_entry

    if num_entries == 0:
        print("No entries found.")
        return

    avg_entry_length = total_length / num_entries
    avg_masks_per_entry = total_masks / num_entries
    mask_ratio = avg_masks_per_entry / avg_entry_length

    print(f"\nNumber of entries: {num_entries}")
    print(f"Average input length (tokens): {avg_entry_length:.2f}")
    print(f"Average masks per entry: {avg_masks_per_entry:.2f}")
    print(f"Average mask ratio: {mask_ratio * 100:.2f}% of tokens are [MASK]")
    print(f"Average masks per entry: {total_masks / num_entries:.2f}")
    print(
        f"Minimum consecutive mask cluster: {min_consecutive if min_consecutive != float('inf') else 0}"
    )
    print(f"Maximum consecutive mask cluster: {max_consecutive}")
    print("\nMask cluster frequencies (mean number of clusters per entry):")
    for cluster_size in sorted(cluster_freq):
        avg_freq = cluster_freq[cluster_size] / num_entries
        print(f"  {cluster_size}x [MASK]: {avg_freq:.2f} per entry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze mask statistics in masked LM training data."
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="HuggingFace tokenizer name or path.",
    )

    args = parser.parse_args()
    analyze_mask_clusters(args.input_file, args.tokenizer)
