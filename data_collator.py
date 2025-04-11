from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


def to_list(tensor_or_iterable):
    if isinstance(tensor_or_iterable, torch.Tensor):
        return tensor_or_iterable.tolist()
    return list(tensor_or_iterable)


@dataclass
class DataCollatorForTagging:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = (
        0  # cannot be -100 because edit_tags are used in forward pass
    )
    return_tensors: str = "pt"

    def __call__(self, features):
        label_names = ["edit_tags", "pointers", "labels_mask"]

        labels = [None] * len(label_names)
        for i, label_name in enumerate(label_names):
            labels[i] = (
                [feature[label_name] for feature in features]
                if label_name in features[0].keys()
                else None
            )

        no_labels_features = [
            {k: v for k, v in feature.items() if k not in label_names}
            for feature in features
        ]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        for i, label_name in enumerate(label_names):
            if padding_side == "right":
                batch[label_name] = [
                    to_list(label) + [0] * (sequence_length - len(label))
                    for label in labels[i]
                ]
            else:
                batch[label_name] = [
                    [0] * (sequence_length - len(label)) + to_list(label)
                    for label in labels[i]
                ]
            batch[label_name] = torch.tensor(
                batch[label_name],
                dtype=torch.float32 if label_name.endswith("_mask") else torch.int64,
            )

        return batch


@dataclass
class DataCollatorForJointModel:
    """
    Data collator for joint models that dynamically pads inputs and labels.
    Handles separate inputs for tagging and insertion tasks.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = (
        0  # Cannot be -100 because edit_tags are used in forward pass
    )
    return_tensors: str = "pt"

    def __call__(self, features):
        # Pad tagging fields
        tagging_features = []
        for feature in features:
            tagging_feature = {
                "input_ids": feature["tagging_input_ids"],
                "attention_mask": feature["tagging_input_mask"],
            }
            if "tagging_token_type_ids" in feature:
                tagging_feature["token_type_ids"] = feature["tagging_token_type_ids"]
            tagging_features.append(tagging_feature)

        tagging_batch = self.tokenizer.pad(
            tagging_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # Pad insertion fields
        insertion_features = []
        for feature in features:
            insertion_feature = {
                "input_ids": feature["insertion_input_ids"],
                "attention_mask": feature["insertion_attention_mask"],
            }
            if "insertion_token_type_ids" in feature:
                insertion_feature["token_type_ids"] = feature[
                    "insertion_token_type_ids"
                ]
            insertion_features.append(insertion_feature)

        insertion_batch = self.tokenizer.pad(
            insertion_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # Rename keys to avoid conflicts
        tagging_batch = {f"tagging_{k}": v for k, v in tagging_batch.items()}
        insertion_batch = {f"insertion_{k}": v for k, v in insertion_batch.items()}

        # Pad other fields (edit_tags, pointers, labels_mask, masked_lm_ids)
        batch = {**tagging_batch, **insertion_batch}
        for field in ["edit_tags", "pointers", "labels_mask", "masked_lm_ids"]:
            if field in features[0]:
                batch[field] = self._pad_labels(
                    [feature[field] for feature in features],
                    sequence_length=(
                        tagging_batch["tagging_input_ids"].shape[1]
                        if field != "masked_lm_ids"
                        else insertion_batch["insertion_input_ids"].shape[1]
                    ),
                    dtype=torch.float32 if field.endswith("_mask") else torch.long,
                )

        return batch

    def _pad_labels(self, labels, sequence_length, dtype=torch.long):
        """
        Helper function to pad labels to the specified sequence length.
        """
        padding_side = self.tokenizer.padding_side
        padded_labels = []
        for label in labels:
            if padding_side == "right":
                padded_label = to_list(label) + [self.label_pad_token_id] * (
                    sequence_length - len(label)
                )
            else:
                padded_label = [self.label_pad_token_id] * (
                    sequence_length - len(label)
                ) + to_list(label)
            padded_labels.append(padded_label)
        if any([len(a) != len(padded_labels[0]) for a in padded_labels]):
            raise ValueError("All labels must have the same length after padding")
        return torch.tensor(padded_labels, dtype=dtype)
