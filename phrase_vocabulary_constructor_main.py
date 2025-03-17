import argparse
import json
import logging
import os


def main(args):
    # Always use these entries.
    label_map = {"PAD": 0, "SWAP": 1, "KEEP": 2, "DELETE": 3}
    # Create Insert 1 MASK to insertion N MASKS.
    for i in range(1, args.max_mask + 1):
        label_map[f"KEEP|{i}"] = len(label_map)
        if not args.use_pointing:
            label_map[f"DELETE|{i}"] = len(label_map)
    logging.info("Created new label map with %d labels", len(label_map))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, mode="w") as f:
        json.dump(label_map, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a label map file for Felix and FelixPointer."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the resulting new label_map_file.",
    )
    parser.add_argument(
        "--max_mask",
        type=int,
        default=16,
        help="The maximum number of MASKs the model can create per input token when `use_open_vocab == True`.",
    )
    parser.add_argument(
        "--use_pointing",
        action="store_true",
        help="If true a pointing mechanism be used. Only True is currently supported.",
    )
    args = parser.parse_args()

    main(args)
