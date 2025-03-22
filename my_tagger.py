import math
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel


class TagLoss(nn.Module):
    def __init__(self, use_pointing=True, pointing_weight=1.0):
        super(TagLoss, self).__init__()
        self.use_pointing = use_pointing
        self.pointing_weight = pointing_weight

    def forward(
        self,
        tag_logits,
        tag_labels,
        attention_mask,
        labels_mask,
        point_logits=None,
        point_labels=None,
    ):
        tag_loss_fct = nn.CrossEntropyLoss(reduction="none")
        tag_logits_loss = tag_loss_fct(
            tag_logits.view(-1, tag_logits.size(-1)),
            tag_labels.view(-1),
        )
        tag_logits_loss = tag_logits_loss * labels_mask.view(-1)
        # Calculate the mean of the masked loss
        tag_logits_loss = tag_logits_loss.mean()

        if self.use_pointing:
            point_loss_fct = nn.CrossEntropyLoss(reduction="none")
            point_logits_loss = point_loss_fct(
                point_logits.view(-1, point_logits.size(-1)),
                point_labels.view(-1),
            )
            point_logits_loss = point_logits_loss * attention_mask.view(-1)
            point_logits_loss = point_logits_loss.mean()

            return tag_logits_loss + self.pointing_weight * point_logits_loss

        return tag_logits_loss


def get_mask(inputs, to_mask):
    """Gets a 3D self-attention mask.

    Args:
    inputs: from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    to_mask = deepcopy(to_mask)
    from_shape = inputs.size()
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = to_mask.size()
    to_seq_length = to_shape[1]

    to_mask = to_mask.view(batch_size, 1, to_seq_length)

    return to_mask.expand(batch_size, from_seq_length, to_seq_length)


class MyTagger(PreTrainedModel):
    """Felix tagger model based on a transformer-based encoder.

    It adds an edit tagger (classification) and optionally a pointer network
    (self-attention) to a Transformer encoder.
    """

    def __init__(
        self,
        config,
        model_name_or_path,
        seq_length=128,
        pointing_weight=1.0,
    ):
        """Creates Felix Tagger.

        Setting up all of the layers needed for call.

        Args:
            network: An encoder network, which should output a sequence of hidden states.
            config: A config file which in addition to the base config values also includes:
                num_classes, hidden_dropout_prob, and query_transformer.
            seq_length: Maximum sequence length.
            use_pointing: Whether a pointing network is used.
        """
        super(MyTagger, self).__init__(config)

        self._backbone = AutoModel.from_pretrained(model_name_or_path, config=config)
        self._seq_length = seq_length
        self._config = config
        self._use_pointing = config.pointing
        self._pointing_weight = pointing_weight

        self._tag_logits_layer = nn.Linear(
            self._config.hidden_size, self._config.num_classes
        )
        if not self._use_pointing:
            return

        # An arbitrary heuristic (sqrt vocab size) for the tag embedding dimension.
        tag_size = int(math.ceil(math.sqrt(self._config.num_classes)))

        self._tag_embedding_layer = nn.Embedding(self._config.num_classes, tag_size)

        self._position_embedding_layer = PositionEmbedding(
            seq_length, embedding_dim=tag_size
        )
        self._edit_tagged_sequence_output_layer = nn.Linear(
            self._config.hidden_size + 2 * tag_size,
            self._config.hidden_size,
        )

        if self._config.query_transformer:
            self._transformer_query_layer = nn.TransformerEncoderLayer(
                d_model=self._config.hidden_size,
                nhead=self._config.num_attention_heads,
                dim_feedforward=self._config.intermediate_size,
                dropout=getattr(
                    self._config,
                    "hidden_dropout_prob",
                    getattr(self._config, "attention_dropout", 0.1),
                ),  # Fallback to attention_dropout or default 0.1
                activation="gelu",
                batch_first=True,
            )

        self._query_embeddings_layer = nn.Linear(
            self._config.hidden_size, self._config.query_size
        )
        self._key_embeddings_layer = nn.Linear(
            self._config.hidden_size, self._config.query_size
        )

    def _attention_scores(self, query, key, mask=None):
        """Calculates attention scores as a query-key dot product.

        Args:
            query: Query tensor of shape `[batch_size, sequence_length, Tq]`.
            key: Key tensor of shape `[batch_size, sequence_length, Tv]`.
            mask: Mask tensor of shape `[batch_size, sequence_length]`.

        Returns:
            Tensor of shape `[batch_size, sequence_length, sequence_length]`.
        """

        scores = torch.matmul(query, key.transpose(1, 2))

        if mask is not None:
            mask = get_mask(scores, mask)
            diagonal_mask = (
                torch.eye(mask.size(1)).unsqueeze(0).repeat(mask.size(0), 1, 1)
            ).to(scores.device)
            diagonal_mask = (1.0 - diagonal_mask).to(torch.float32)
            mask = diagonal_mask * mask
            mask_add = -1e9 * (1.0 - mask)
            scores = scores * mask + mask_add

        return scores

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        edit_tags=None,
        pointers=None,
        labels_mask=None,
    ):
        """Forward pass of the model.

        Args:
            inputs: A list of tensors. In training, the following 4 tensors are required:
                [input_word_ids, attention_mask, input_type_ids, edit_tags].
                Only the first 3 are required in test.
                input_word_ids [batch_size, seq_length],
                attention_mask [batch_size, seq_length],
                input_type_ids [batch_size, seq_length],
                edit_tags [batch_size, seq_length].
                If using output variants, these should also be provided.
                output_variant_ids [batch_size, 1].

        Returns:
            The logits of the edit tags and optionally the logits of the pointer network.
        """
        backbone_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        if token_type_ids is not None:
            backbone_inputs["token_type_ids"] = token_type_ids

        backbone_output = self._backbone(**backbone_inputs)[0]
        tag_logits = self._tag_logits_layer(backbone_output)

        if not self._use_pointing:
            return [tag_logits]

        if not self.training:
            edit_tags = torch.argmax(tag_logits, dim=-1)

        tag_embedding = self._tag_embedding_layer(edit_tags)
        position_embedding = self._position_embedding_layer(tag_embedding)

        edit_tagged_sequence_output = self._edit_tagged_sequence_output_layer(
            torch.cat([backbone_output, tag_embedding, position_embedding], dim=-1)
        )

        intermediate_query_embeddings = edit_tagged_sequence_output
        if self._config.query_transformer:
            query_mask = get_mask(intermediate_query_embeddings, attention_mask)
            intermediate_query_embeddings = self._transformer_query_layer(
                intermediate_query_embeddings,
                query_mask[0].float(),
            )

        query_embeddings = self._query_embeddings_layer(intermediate_query_embeddings)
        key_embeddings = self._key_embeddings_layer(edit_tagged_sequence_output)

        pointing_logits = self._attention_scores(
            query_embeddings, key_embeddings, attention_mask.float()
        )

        if self.training and pointers is not None and labels_mask is not None:
            loss_fct = TagLoss(self._use_pointing, self._pointing_weight)
            loss = loss_fct(
                tag_logits,
                edit_tags,
                attention_mask,
                labels_mask,
                pointing_logits,
                pointers,
            )
            return loss, tag_logits, pointing_logits
        else:
            return tag_logits, pointing_logits


class PositionEmbedding(nn.Module):
    def __init__(self, max_length, embedding_dim, seq_axis=1):
        super(PositionEmbedding, self).__init__()
        if max_length is None:
            raise ValueError("`max_length` must be an Integer, not `None`.")

        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.seq_axis = seq_axis

        # Define position embeddings as a learnable parameter
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, max_length, embedding_dim)
        )
        nn.init.xavier_uniform_(self.position_embeddings)

    def forward(self, inputs):
        input_shape = inputs.size()
        batch_size = input_shape[0]  # Get batch size dynamically
        actual_seq_len = input_shape[self.seq_axis]

        # Slice the required embeddings
        position_embeddings = self.position_embeddings[:, :actual_seq_len, :]

        # Expand to match batch size
        return position_embeddings.expand(
            batch_size, actual_seq_len, self.embedding_dim
        )
