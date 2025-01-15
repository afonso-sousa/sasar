from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


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

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

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
