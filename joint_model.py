import math

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    PretrainedConfig,
    PreTrainedModel,
)

from my_tagger import PositionEmbedding, TagLoss, get_mask


class JointModelConfig(PretrainedConfig):
    model_type = "jointmodel"

    def __init__(
        self,
        pointing=True,
        num_classes=10,
        query_size=64,
        query_transformer=True,
        pointing_weight=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pointing = pointing
        self.num_classes = num_classes
        self.query_size = query_size
        self.query_transformer = query_transformer
        self.pointing_weight = pointing_weight


class JointModel(PreTrainedModel):

    config_class = JointModelConfig

    def __init__(self, config):
        super().__init__(config)

        encoder_config = AutoConfig.from_pretrained(config.backbone_name)
        encoder_config.vocab_size = config.vocab_size
        self.encoder = AutoModel.from_pretrained(
            config.backbone_name, config=encoder_config, ignore_mismatched_sizes=True
        )

        # Check if the encoder supports token_type_ids
        self.supports_token_type_ids = (
            hasattr(self.encoder.config, "type_vocab_size")
            and self.encoder.config.type_vocab_size > 1
        )

        mlm_aux_model = AutoModelForMaskedLM.from_pretrained(
            config.backbone_name, config=encoder_config, ignore_mismatched_sizes=True
        )
        if hasattr(mlm_aux_model, "cls"):  # For models like BERT
            self.my_mlm_head = mlm_aux_model.cls
        else:  # For models like ModernBert
            self.my_head = mlm_aux_model.head
            self.my_decoder = mlm_aux_model.decoder

        self.seq_length = config.seq_length
        self.use_pointing = config.pointing
        self.pointing_weight = config.pointing_weight

        self.tag_logits_layer = nn.Linear(config.hidden_size, config.num_classes)

        if self.use_pointing:
            tag_size = int(math.ceil(math.sqrt(config.num_classes)))
            self.tag_embedding_layer = nn.Embedding(config.num_classes, tag_size)
            self.position_embedding_layer = PositionEmbedding(
                self.seq_length, embedding_dim=tag_size
            )
            self.edit_tagged_sequence_output_layer = nn.Linear(
                config.hidden_size + 2 * tag_size, config.hidden_size
            )
            self.query_embeddings_layer = nn.Linear(
                config.hidden_size, config.query_size
            )
            self.key_embeddings_layer = nn.Linear(config.hidden_size, config.query_size)

            if hasattr(config, "with_sinkhorn"):
                from sinkhorn_layer import Sinkhorn

                self.sinkhorn = Sinkhorn()

    def get_tagging_predictions(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        edit_tags=None,
    ):
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.supports_token_type_ids and token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        tagging_outputs = self.encoder(**encoder_kwargs)[0]
        tag_logits = self.tag_logits_layer(tagging_outputs)

        if not self.use_pointing:
            return tag_logits, None

        if not self.training:
            edit_tags = torch.argmax(tag_logits, dim=-1)

        tag_embedding = self.tag_embedding_layer(edit_tags)
        position_embedding = self.position_embedding_layer(tag_embedding)
        edit_tagged_sequence_output = self.edit_tagged_sequence_output_layer(
            torch.cat([tagging_outputs, tag_embedding, position_embedding], dim=-1)
        )

        query_embeddings = self.query_embeddings_layer(edit_tagged_sequence_output)
        key_embeddings = self.key_embeddings_layer(edit_tagged_sequence_output)
        pointing_logits = self._attention_scores(
            query_embeddings, key_embeddings, attention_mask.float()
        )

        if hasattr(self, "sinkhorn"):
            pointing_logits = self.sinkhorn(pointing_logits)

        return tag_logits, pointing_logits

    def get_insertion_predictions(self, input_ids, attention_mask, token_type_ids=None):
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.supports_token_type_ids and token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        insertion_outputs = self.encoder(**encoder_kwargs)[0]

        if hasattr(self, "my_mlm_head"):  # For models like BERT
            return self.my_mlm_head(insertion_outputs)
        else:  # For models like ModernBert
            return self.my_decoder(self.my_head(insertion_outputs))

    def forward(
        self,
        tagging_input_ids,
        tagging_attention_mask,
        insertion_input_ids,
        insertion_attention_mask,
        tagging_token_type_ids=None,
        insertion_token_type_ids=None,
        edit_tags=None,
        pointers=None,
        labels_mask=None,
        masked_lm_ids=None,
    ):

        tag_logits, pointing_logits = self.get_tagging_predictions(
            tagging_input_ids,
            tagging_attention_mask,
            tagging_token_type_ids,
            edit_tags,
        )

        mlm_logits = self.get_insertion_predictions(
            insertion_input_ids, insertion_attention_mask, insertion_token_type_ids
        )

        if not self.training:
            return (tag_logits, pointing_logits, mlm_logits)

        mlm_loss = None
        if masked_lm_ids is not None:
            loss_fct = nn.CrossEntropyLoss()

            mlm_loss = loss_fct(
                mlm_logits.view(-1, mlm_logits.size(-1)), masked_lm_ids.view(-1)
            )

        if not self.use_pointing:
            total_loss = mlm_loss if mlm_loss is not None else None
            return (
                (total_loss, tag_logits, mlm_logits)
                if self.training
                else (tag_logits, mlm_logits)
            )

        # Tagging Loss
        tagging_loss = None
        if self.training and pointers is not None and labels_mask is not None:
            loss_fct = TagLoss(self.use_pointing, self.pointing_weight)
            tagging_loss = loss_fct(
                tag_logits,
                edit_tags,
                tagging_attention_mask,
                labels_mask,
                pointing_logits,
                pointers,
            )

        # Combined Loss
        total_loss = None
        if mlm_loss is not None and tagging_loss is not None:
            total_loss = mlm_loss + tagging_loss
        elif mlm_loss is not None:
            total_loss = mlm_loss
        elif tagging_loss is not None:
            total_loss = tagging_loss

        return total_loss, tag_logits, pointing_logits, mlm_logits

    def _attention_scores(self, query, key, mask=None):
        scores = torch.matmul(query, key.transpose(1, 2))
        if mask is not None:
            mask = get_mask(scores, mask)
            diagonal_mask = (
                1.0 - torch.eye(mask.size(1), device=scores.device)
            ).unsqueeze(0)
            mask = diagonal_mask * mask
            mask_add = -1e9 * (1.0 - mask)
            scores = scores * mask + mask_add
        return scores
