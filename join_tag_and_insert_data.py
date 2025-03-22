import json
import os


def join_tagging_and_insertion_data(
    tagging_data_path: str, insertion_data_path: str, output_path: str
):
    """
    Joins tagging and insertion data into a single dataset for joint model training.
    Keeps input_ids and input_mask separate for tagging and insertion tasks.

    Args:
        tagging_data_path (str): Path to the tagging data file (JSONL format).
        insertion_data_path (str): Path to the insertion data file (JSONL format).
        output_path (str): Path to save the joined dataset (JSONL format).
    """
    # Load tagging data
    with open(tagging_data_path, "r", encoding="utf-8") as f:
        tagging_data = [json.loads(line) for line in f]

    # Load insertion data
    with open(insertion_data_path, "r", encoding="utf-8") as f:
        insertion_data = [json.loads(line) for line in f]

    # Ensure both datasets have the same length
    if len(tagging_data) != len(insertion_data):
        raise ValueError(
            "Tagging and insertion datasets must have the same number of examples."
        )

    # Combine the datasets
    joint_data = []
    for tagging_example, insertion_example in zip(tagging_data, insertion_data):
        # Create a joint example with separate fields for tagging and insertion
        joint_example = {
            # Tagging task fields
            "tagging_input_ids": tagging_example["input_ids"],
            "tagging_input_mask": tagging_example["input_mask"],
            "tagging_token_type_ids": tagging_example["token_type_ids"],
            "point_indexes": tagging_example["point_indexes"],
            "labels": tagging_example["labels"],
            "labels_mask": tagging_example["labels_mask"],
            # Insertion task fields
            "insertion_input_ids": insertion_example["input_ids"],
            "insertion_input_mask": insertion_example["input_mask"],
            "insertion_token_type_ids": insertion_example["token_type_ids"],
            "masked_lm_ids": insertion_example["masked_lm_ids"],
        }
        joint_data.append(joint_example)

    # Save the joined dataset
    with open(output_path, "w", encoding="utf-8") as f:
        for example in joint_data:
            f.write(json.dumps(example) + "\n")

    print(f"Joined dataset saved to {output_path}")


# Example usage
if __name__ == "__main__":
    input_dir = "input/paws"
    split = "train"
    with_graph = True
    with_deleted_spans = False

    core_name = f'{split}{"_with_graph" if with_graph else ""}{"_include_del_spans" if with_deleted_spans else "_no_del_spans"}'
    tagging_data_path = os.path.join(input_dir, f"{core_name}.json")
    insertion_data_path = os.path.join(input_dir, f"{core_name}.json.ins")
    output_path = os.path.join(input_dir, f"{core_name}_joint.jsonl")
    join_tagging_and_insertion_data(tagging_data_path, insertion_data_path, output_path)
