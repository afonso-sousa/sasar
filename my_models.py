"""Defines models."""

from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

import joint_model
from my_tagger import MyTagger, MyTaggerConfig

AutoConfig.register("mytagger", MyTaggerConfig)
AutoModel.register(MyTaggerConfig, MyTagger)


def get_insertion_model(config, model_name_or_path):
    model = AutoModelForMaskedLM.from_pretrained(
        model_name_or_path, config=config, ignore_mismatched_sizes=True
    )
    return model


def get_tagging_model(
    model_name_or_path,
    seq_length,
    pointing_weight=1.0,
    use_pointing=False,
    num_classes=10,
    vocab_size=30522,
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
    config = MyTaggerConfig.from_pretrained(model_name_or_path)
    config.backbone_name = model_name_or_path
    config.pointing = use_pointing
    config.pointing_weight = pointing_weight
    config.seq_length = seq_length
    config.num_classes = num_classes
    config.query_size = 64
    config.query_transformer = True
    config.vocab_size = vocab_size
    model = MyTagger(config)

    return model


def get_joint_model(
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

    model = joint_model.JointModel(
        config=config,
        model_name_or_path=model_name_or_path,
        seq_length=seq_length,
        pointing_weight=pointing_weight,
    )

    return model
