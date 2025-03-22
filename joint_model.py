import math

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForMaskedLM, PreTrainedModel

from my_tagger import PositionEmbedding, TagLoss, get_mask


class JointModel(PreTrainedModel):
    def __init__(self, config, model_name_or_path, seq_length=128, pointing_weight=1.0):
        super(JointModel, self).__init__(config)

        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=config)

        # Check if the encoder supports token_type_ids
        self.supports_token_type_ids = (
            hasattr(self.encoder.config, "type_vocab_size")
            and self.encoder.config.type_vocab_size > 1
        )

        mlm_aux_model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, config=config
        )
        if hasattr(mlm_aux_model, "cls"):  # For models like BERT
            self.mlm_head = mlm_aux_model.cls
        else:  # For models like ModernBert
            self.head = mlm_aux_model.head
            self.decoder = mlm_aux_model.decoder

        self.seq_length = seq_length
        self.use_pointing = config.pointing
        self.pointing_weight = pointing_weight

        self.tag_logits_layer = nn.Linear(config.hidden_size, config.num_classes)

        if self.use_pointing:
            tag_size = int(math.ceil(math.sqrt(config.num_classes)))
            self.tag_embedding_layer = nn.Embedding(config.num_classes, tag_size)
            self.position_embedding_layer = PositionEmbedding(
                seq_length, embedding_dim=tag_size
            )
            self.edit_tagged_sequence_output_layer = nn.Linear(
                config.hidden_size + 2 * tag_size, config.hidden_size
            )
            self.query_embeddings_layer = nn.Linear(
                config.hidden_size, config.query_size
            )
            self.key_embeddings_layer = nn.Linear(config.hidden_size, config.query_size)

    def forward(
        self,
        tagging_input_ids,
        tagging_attention_mask,
        tagging_token_type_ids=None,
        insertion_input_ids=None,
        insertion_attention_mask=None,
        insertion_token_type_ids=None,
        edit_tags=None,
        pointers=None,
        labels_mask=None,
        masked_lm_ids=None,
    ):
        # Tagging task
        encoder_kwargs = {
            "input_ids": tagging_input_ids,
            "attention_mask": tagging_attention_mask,
        }
        if self.supports_token_type_ids and tagging_token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = tagging_token_type_ids
        tagging_outputs = self.encoder(**encoder_kwargs)[0]

        # Tagging Head
        tag_logits = self.tag_logits_layer(tagging_outputs)

        # Insertion task
        encoder_kwargs = {
            "input_ids": insertion_input_ids,
            "attention_mask": insertion_attention_mask,
        }
        if self.supports_token_type_ids and insertion_token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = insertion_token_type_ids
        insertion_outputs = self.encoder(**encoder_kwargs)[0]

        # MLM Loss
        if hasattr(self, "mlm_head"):  # For models like BERT
            mlm_logits = self.mlm_head(insertion_outputs)
        else:  # For models like ModernBert
            mlm_logits = self.decoder(self.head(insertion_outputs))

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
            query_embeddings, key_embeddings, tagging_attention_mask.float()
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

        return (
            (total_loss, tag_logits, pointing_logits, mlm_logits)
            if self.training
            else (tag_logits, pointing_logits, mlm_logits)
        )

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
