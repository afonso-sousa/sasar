"""Defines models."""

from transformers import BertConfig, BertForMaskedLM

import my_tagger
from utils import are_configurations_equal


def get_insertion_model(bert_config):
    is_base_config = are_configurations_equal(
        bert_config,
        BertConfig(),
    )

    if is_base_config:
        model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    else:
        model = BertForMaskedLM(config=bert_config)

    return model


def get_tagging_model(
    bert_config,
    seq_length,
    pointing_weight=1.0,
):
    """Returns model to be used for pre-training.

    Args:
        bert_config: Configuration that defines the core BERT model.
        seq_length: Maximum sequence length of the training data.
        use_pointing: If FELIX should use a pointer (reordering) model.
        pointing_weight: How much to weigh the pointing loss, in contrast to
          tagging loss. Note, if pointing is set to false this is ignored.

    Returns:
        Felix model as well as core BERT submodel from which to save
        weights after pretraining.
    """

    # Check if the provided bert_config is the same as the base BERT configuration

    model = my_tagger.MyTagger(
        config=bert_config,
        seq_length=seq_length,
        pointing_weight=pointing_weight,
    )

    return model
