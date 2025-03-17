"""Defines models."""

from transformers import AutoModelForMaskedLM

import my_tagger


def get_insertion_model(config, model_name_or_path):
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config)
    return model


def get_tagging_model(
    config,
    model_name_or_path,
    seq_length,
    pointing_weight=1.0,
):
    """Returns model to be used for pre-training.

    Args:
        config: Configuration that defines the core model.
        seq_length: Maximum sequence length of the training data.
        use_pointing: If FELIX should use a pointer (reordering) model.
        pointing_weight: How much to weigh the pointing loss, in contrast to
          tagging loss. Note, if pointing is set to false this is ignored.

    Returns:
        Felix model as well as core Transformer submodel from which to save
        weights after pretraining.
    """

    model = my_tagger.MyTagger(
        config=config,
        model_name_or_path=model_name_or_path,
        seq_length=seq_length,
        pointing_weight=pointing_weight,
    )

    return model
