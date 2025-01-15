"""Utility functions for Felix."""

import json

from datasets import Dataset, load_dataset

import constants


def build_feed_dict(
    tokens,
    tokenizer,
    target_tokens=None,
    max_seq_length=128,
    label_pad_token_id=-100,
):
    """Returns a dictionary used for predicting/training the insertion model.

    Converts a list of source tokens, containing masks, to a dictionary of
    features used by a PyTorch model. If a target sequence is provided, then the targets for the MASKs are set.

    Args:
      tokens: Input tokens, with mask tokens.
      tokenizer: Tokenizer used to convert tokens to IDs.
      target_tokens: (Optional) The targets of the mask tokens.
      max_seq_length: Maximum sequence length.

    Returns:
      Dictionary with model features or None if `len(tokens) > max_seq_length` or if the number of MASKs is larger than `max_predictions_per_seq`.
    """

    # Deleted tokens (bracketed by unused) should have a segment_id of 2.
    unused = False
    token_type_ids = []
    for token in tokens:
        if token == constants.DELETE_SPAN_START or unused:
            unused = True
            token_type_ids.append(1)
        else:
            token_type_ids.append(0)
        if token == constants.DELETE_SPAN_END:
            unused = False
    input_mask = [1] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    assert len(token_type_ids) == len(input_ids)

    if len(input_ids) > max_seq_length:
        return None

    feed_dict = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "token_type_ids": token_type_ids,
    }

    if target_tokens:
        mask_target_id = []
        for idx, token in enumerate(tokens):
            if token == tokenizer.mask_token:
                mask_target_id.append(
                    tokenizer.convert_tokens_to_ids(target_tokens[idx])
                )
            else:
                mask_target_id.append(label_pad_token_id)

            if token != tokenizer.mask_token:
                continue

        assert len(mask_target_id) == len(input_ids)

        feed_dict["masked_lm_ids"] = mask_target_id

    return feed_dict


def yield_sources_and_targets(
    dataset_or_name, split="train", source_key=None, target_key=None
):
    """Produces an iterator over pairs (source list, targets) parsed from a file.

    Args:
      input_file_pattern: Path/pattern to the input file(s).
      input_format: Format of the input file.
      source_key: Source text feature name. Only considered when
        `input_format=sstable`.
      target_key: Target text feature name. Only considered when
        `input_format=sstable`.

    Yields:
      Pairs of (list of source texts, target text).
    """
    if isinstance(dataset_or_name, str):
        if dataset_or_name == "paws":
            dataset = load_dataset(dataset_or_name, "labeled_final", split=split)
            dataset = dataset.filter(lambda x: x["label"] == 1)
            dataset = dataset.rename_column("sentence1", "source")
            dataset = dataset.rename_column("sentence2", "target")
        else:
            dataset = load_dataset(dataset_or_name, split=split)
    elif isinstance(dataset_or_name, Dataset):
        dataset = dataset_or_name
    else:
        raise ValueError(
            "`dataset_or_name` must be a string or a Hugging Face Dataset object."
        )

    for item in dataset:
        source_texts = item[source_key] if source_key else item["source"]
        target_text = item[target_key] if target_key else item["target"]
        if source_texts is not None and target_text is not None:
            if isinstance(source_texts, str):
                source_texts = [source_texts]
            yield source_texts, target_text


def read_label_map(path, use_str_keys=False):
    """Returns label map read from the given path.

    Args:
      path: Path to the label map file.
      use_str_keys: Whether to use label strings as keys instead of
        (base tag, num insertions) tuple keys. The latter is only used by
        FelixInsert.
    """
    label_map = {}
    with open(path, mode="r") as f:
        label_map = json.load(f)
    if not use_str_keys:
        new_label_map = {}
        for key, val in label_map.items():
            if "|" in key:
                pos_pipe = key.index("|")
                new_key = (key[:pos_pipe], int(key[pos_pipe + 1 :]))
            else:
                new_key = (key, 0)
            new_label_map[new_key] = val
        label_map = new_label_map
    return label_map


def are_configurations_equal(config1, config2):
    """Check if the common keys have the same values in two configuration objects."""
    common_keys = set(config1.__dict__.keys()).intersection(config2.__dict__.keys())
    for key in common_keys:
        if getattr(config1, key) != getattr(config2, key):
            return False
    return True
