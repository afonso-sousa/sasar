import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForMaskedLM, PreTrainedModel

from my_tagger import PositionEmbedding, TagLoss, get_mask


class JointModel(PreTrainedModel):
    def __init__(self, config, model_name_or_path, seq_length=128, pointing_weight=1.0):
        super(JointModel, self).__init__(config)

        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.mlm_head = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, config=config
        ).lm_head

        self.seq_length = seq_length
        self.use_pointing = config.pointing
        self.pointing_weight = pointing_weight

        self.tag_logits_layer = nn.Linear(config.hidden_size, config.num_classes)

        if self.use_pointing:
            tag_size = int(config.num_classes**0.5)
            self.tag_embedding_layer = nn.Embedding(config.num_classes, tag_size)
            self.position_embedding_layer = PositionEmbedding(seq_length)
            self.edit_tagged_sequence_output_layer = nn.Linear(
                config.hidden_size + 2 * tag_size, config.hidden_size
            )
            self.query_embeddings_layer = nn.Linear(
                config.hidden_size, config.query_size
            )
            self.key_embeddings_layer = nn.Linear(config.hidden_size, config.query_size)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        edit_tags=None,
        pointers=None,
        labels_mask=None,
        mlm_labels=None,
    ):
        backbone_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]

        # MLM Loss
        mlm_logits = self.mlm_head(backbone_outputs)
        mlm_loss = None
        if mlm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            mlm_loss = loss_fct(
                mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1)
            )

        # Tagging Head
        tag_logits = self.tag_logits_layer(backbone_outputs)

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
            torch.cat([backbone_outputs, tag_embedding, position_embedding], dim=-1)
        )

        query_embeddings = self.query_embeddings_layer(edit_tagged_sequence_output)
        key_embeddings = self.key_embeddings_layer(edit_tagged_sequence_output)
        pointing_logits = self._attention_scores(
            query_embeddings, key_embeddings, attention_mask.float()
        )

        # Tagging Loss
        tagging_loss = None
        if self.training and pointers is not None and labels_mask is not None:
            loss_fct = TagLoss(self.use_pointing, self.pointing_weight)
            tagging_loss = loss_fct(
                tag_logits,
                edit_tags,
                attention_mask,
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
