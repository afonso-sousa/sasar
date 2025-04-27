import os

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    DataCollatorForSeq2Seq,
)

import beam_search
import constants
import insertion_converter
import preprocess
import utils
from data_collator import DataCollatorForJointModel
from joint_model import JointModel, JointModelConfig
from my_tagger import MyTagger, MyTaggerConfig

AutoConfig.register("mytagger", MyTaggerConfig)
AutoModel.register(MyTaggerConfig, MyTagger)

AutoConfig.register("jointmodel", JointModelConfig)
AutoModel.register(JointModelConfig, JointModel)


class Predictor:
    def __init__(
        self,
        model_tagging_filepath,
        model_insertion_filepath,
        tokenizer_name,
        label_map_file,
        sequence_length=128,
        use_open_vocab=False,
        is_pointing=False,
        special_glue_string_for_joining_sources=" ",
        use_token_type_ids=False,
        no_deleted_spans=False,
    ):
        """Initializes an instance of FelixPredictor.

        Args:
            model_tagging_filepath: The file location of the tagging model.  If not
              provided in a randomly initialized model is used (not recommended).
            model_insertion_filepath: The file location of the insertion model. Only
              needed in the case of use_open_vocab (Felix/FelixInsert)  and if not
              provided  a randomly initialized model is used (not recommended).
            label_map_file: Label map file path.
            sequence_length: Maximum length of a sequence.
            do_lowercase: If the input text is lowercased.
            use_open_vocab: If it is an open vocab model (felix/felixInsert).
              Currently only True is supported.
            is_pointing: if the tagging model uses a pointing mechanism. Currently
              only True is supported.
            insert_after_token: Whether to insert tokens after rather than before
              the current token.
            special_glue_string_for_joining_sources: String that is used to join multiple source strings of a given example into one string.
            use_token_type_ids: Whether to use token type ids in the model.
        """
        if not use_open_vocab:
            raise ValueError("Currently only use_open_vocab=True is supported")
        if model_tagging_filepath is None:
            print(
                "No filepath is provided for tagging model, a randomly initialized "
                "model will be used!"
            )
        if model_insertion_filepath is None and not use_open_vocab:
            print(
                "No filepath is provided for insertion model, a randomly initialized model will be used!"
            )
        self._model_tagging_filepath = model_tagging_filepath
        self._model_insertion_filepath = model_insertion_filepath
        self._tagging_model = None
        self._insertion_model = None

        self._sequence_length = sequence_length
        self._use_open_vocab = use_open_vocab
        self._is_pointing = is_pointing
        self._use_token_type_ids = use_token_type_ids

        self._builder = preprocess.initialize_builder(
            use_open_vocab=self._use_open_vocab,
            label_map_file=label_map_file,
            max_seq_length=self._sequence_length,
            tokenizer_name=tokenizer_name,
            special_glue_string_for_sources=special_glue_string_for_joining_sources,
            include_deleted_spans=not no_deleted_spans,
        )
        self._inverse_label_map = {
            tag_id: tag for tag, tag_id in self._builder.label_map.items()
        }
        self._no_deleted_spans = no_deleted_spans

    def predict_end_to_end_batch(self, batch):
        """Takes in a batch of source sentences and runs Felix on them.

        Args:
          batch: Source inputs, where each input is composed of multiple source
            utterances.

        Returns:
          taggings_outputs: Intermediate realized output of the tagging model.
          insertion_outputs: The final realized output of the model after running the tagging and insertion model.
        """
        taggings_outputs = self._predict_and_realize_batch(batch, is_insertion=False)
        insertion_outputs = self._predict_and_realize_batch(
            taggings_outputs, is_insertion=True
        )
        return taggings_outputs, insertion_outputs

    def _load_model(self, is_insertion=True):
        """Loads either an insertion or tagging model for inference."""
        if is_insertion:
            config_path = os.path.join(self._model_insertion_filepath, "config.json")
            config = AutoConfig.from_pretrained(config_path)
            self._insertion_model = AutoModelForMaskedLM.from_pretrained(
                self._model_insertion_filepath,
                config=config,
            )
            self._insertion_model.eval()
        else:
            self._tagging_model = MyTagger.from_pretrained(self._model_tagging_filepath)
            self._tagging_model.eval()

    def _predict_and_realize_batch(self, source_sentences, is_insertion):
        """Run tagging inference on a batch and return the realizations."""
        (
            batch_dictionaries,
            batch_list,
        ) = self._convert_source_sentences_into_batch(
            source_sentences, is_insertion=is_insertion
        )
        predictions = self._predict_batch(batch_list, is_insertion=is_insertion)
        if is_insertion:
            realizations = self._realize_insertion_batch(
                batch_dictionaries, predictions
            )
        else:
            realizations = self._realize_tagging_batch(batch_dictionaries, predictions)
        return realizations

    def _create_insertion_example(self, source_sentence):
        """Creates an insertion example from a source sentence."""
        # Note source_sentence is the output from the tagging model and therefore already tokenized.
        return utils.build_feed_dict(
            source_sentence.split(" "),
            self._builder.tokenizer,
            max_seq_length=self._sequence_length,
        )

    def _convert_source_sentences_into_batch(self, source_sentences, is_insertion):
        """Converts source sentence into a batch."""
        batch_dictionaries = []
        for source_sentence in source_sentences:
            if is_insertion:
                # Note source_sentence is the output from the tagging model and
                # therefore already tokenized.
                example = utils.build_feed_dict(
                    source_sentence.split(" "),
                    self._builder.tokenizer,
                    max_seq_length=self._sequence_length,
                )

                assert example is not None, (
                    f"Source sentence '{source_sentence}' returned None when "
                    "converting to insertion example."
                )

                # Note masked_lm_ids and masked_lm_weights are filled with zeros.
                batch_dict = {
                    "input_ids": torch.tensor(example["input_ids"]),
                    "attention_mask": torch.tensor(example["input_mask"]),
                }
                if "token_type_ids" in example and self._use_token_type_ids:
                    batch_dict["token_type_ids"] = torch.tensor(
                        example["token_type_ids"]
                    )

                batch_dictionaries.append(batch_dict)
            else:
                example, _ = self._builder.build_transformer_example(
                    [source_sentence],
                    target=None,
                    is_test_time=True,
                    use_token_type_ids=self._use_token_type_ids,
                )

                assert example is not None, (
                    f"Tagging could not convert " f"{source_sentence}."
                )
                batch_dict = {
                    "input_ids": torch.tensor(example.input_ids),
                    "attention_mask": torch.tensor(example.input_mask),
                }
                if self._use_token_type_ids:
                    batch_dict["token_type_ids"] = torch.tensor(example.token_type_ids)

                batch_dictionaries.append(batch_dict)

        data_collator = DataCollatorForSeq2Seq(self._builder.tokenizer)

        batch_list = data_collator(batch_dictionaries)

        # DataCollatorForSeq2Seq will add "labels" to the batch_list, which is not needed.
        if "labels" in batch_list:
            del batch_list["labels"]

        if not self._use_token_type_ids and "token_type_ids" in batch_list:
            del batch_list["token_type_ids"]

        return batch_dictionaries, batch_list

    def _predict_batch(self, source_batch, is_insertion):
        """Produce output from pytorch model."""
        if is_insertion:
            if self._insertion_model is None:
                self._load_model(is_insertion=True)
            predictions = self._insertion_model(**source_batch)[0]
            # Go from a probability distribution to a vocab item.
            return torch.argmax(predictions, dim=-1)

        if self._tagging_model is None:
            self._load_model(is_insertion=False)
        if self._is_pointing:
            tag_logits, pointing_logits = self._tagging_model(**source_batch)
            # Convert two lists into a single list of tuples.
            return list(zip(tag_logits, pointing_logits))
        else:
            tag_logits = self._tagging_model(**source_batch)
            return tag_logits

    def _realize_insertion_batch(self, source_batch, prediction_batch):
        """Produces the realized predicitions for a batch from the tagging model."""
        realizations = []
        for source, predicted_tokens in zip(source_batch, prediction_batch):
            sequence_length = sum(source["attention_mask"])
            realization = self._realize_insertion_single(
                source["input_ids"], sequence_length - 1, predicted_tokens
            )
            realizations.append(realization)
        return realizations

    def _realize_insertion_single(self, input_ids, end_index, predicted_tokens):
        """Realizes the predictions from the insertion model."""
        delete_span_end_token_id = self._builder.tokenizer.convert_tokens_to_ids(
            constants.DELETE_SPAN_END
        )
        delete_span_start_token_id = self._builder.tokenizer.convert_tokens_to_ids(
            constants.DELETE_SPAN_START
        )
        current_mask = 0
        new_ids = []
        in_deletion_bracket = False
        for token_id in input_ids:
            if token_id == delete_span_end_token_id:
                in_deletion_bracket = False
                continue
            elif in_deletion_bracket:
                continue
            elif token_id == delete_span_start_token_id:
                in_deletion_bracket = True
                continue

            if token_id == self._builder.tokenizer.mask_token_id:
                if predicted_tokens.ndim == 0:
                    predicted_tokens = predicted_tokens.unsqueeze(0)
                new_ids.append(predicted_tokens[current_mask])
                current_mask += 1
            else:
                new_ids.append(token_id)

        new_tokens = self._builder.tokenizer.decode(new_ids)
        return new_tokens

    def _realize_tagging_batch(self, source_batch, prediction_batch):
        """Produces the realized predictions for a batch from the tagging model."""
        realizations = []
        for source, prediction in zip(source_batch, prediction_batch):
            end_index = sum(source["attention_mask"]) - 1

            if self._is_pointing:
                tag_logits, pointing_logits = prediction
                realization = self._realize_tagging_single(
                    source["input_ids"],
                    end_index,
                    tag_logits,
                    pointing_logits,
                )
            else:
                tag_logits = prediction
                realization = self._realize_tagging_wo_pointing_single(
                    source["input_ids"], end_index, tag_logits
                )
            # Copy source sentence if prediction has failed.
            if realization is None:
                realization = self._builder.tokenizer.convert_ids_to_tokens(
                    source["input_ids"][: end_index + 1]
                )

            realization = " ".join(realization)
            realizations.append(realization)
        return realizations

    def _realize_tagging_single(
        self,
        input_ids,
        last_token_index,
        tag_logits,
        point_logits,
        beam_size=15,
    ):
        """Returns realized prediction for a given source using beam search.

        Args:
          input_ids:  Source token ids.
          last_token_index: The index, in the input_ids, of the last token (not
            including padding tokens).
          tag_logits: Tag logits  [vocab size, sequence_length].
          point_logits: Point logits  [sequence_length, sequence_length] .
          beam_size: The size of the beam.

        Returns:
          Realized predictions including deleted tokens. It is possible that beam search fails (producing malformed output), in this case return None.
        """
        predicted_tags = torch.argmax(tag_logits, dim=1).tolist()
        non_deleted_indexes = set(
            i
            for i, tag in enumerate(predicted_tags[: last_token_index + 1])
            if self._inverse_label_map[int(tag)] not in constants.DELETED_TAGS
        )
        source_tokens = self._builder.tokenizer.convert_ids_to_tokens(list(input_ids))
        sep_indexes = set(
            [
                i
                for i, token in enumerate(source_tokens)
                if token.lower() == self._builder.tokenizer.sep_token.lower()
                and i in non_deleted_indexes
            ]
        )

        best_sequence = beam_search.beam_search_single_tagging(
            point_logits,
            non_deleted_indexes,
            sep_indexes,
            beam_size,
            last_token_index,
            self._sequence_length,
        )
        if best_sequence is None:
            return None

        return self._realize_beam_search(
            input_ids, best_sequence, predicted_tags, last_token_index + 1
        )

    def _realize_beam_search(
        self, source_token_ids, ordered_source_indexes, tags, source_length
    ):
        """Returns realized prediction using indexes and tags.

        TODO: Refactor this function to share code with
        `_create_masked_source` from insertion_converter.py to reduce code
        duplication and to ensure that the insertion example creation is consistent between preprocessing and prediction.

        Args:
          source_token_ids: List of source token ids.
          ordered_source_indexes: The order in which the kept tokens should be
            realized.
          tags: a List of tags.
          source_length: How long is the source input (excluding padding).

        Returns:
          Realized predictions (with deleted tokens).
        """
        source_token_ids = source_token_ids.numpy()
        ordered_source_indexes = ordered_source_indexes.numpy()

        source_token_ids_set = set(ordered_source_indexes)
        out_tokens = []
        out_tokens_with_deletes = []
        for j, index in enumerate(ordered_source_indexes):
            token = self._builder.tokenizer.convert_ids_to_tokens(
                [source_token_ids[index]]
            )
            out_tokens += token
            tag = self._inverse_label_map[tags[index]]
            if self._use_open_vocab:
                out_tokens_with_deletes += token
                # Add the predicted MASK tokens.
                number_of_masks = insertion_converter.get_number_of_masks(tag)
                # Can not add phrases after last token.
                if j == len(ordered_source_indexes) - 1:
                    number_of_masks = 0
                masks = [constants.MASK] * number_of_masks
                out_tokens += masks
                out_tokens_with_deletes += masks

                # Only include deleted tokens if no_deleted_spans is False
                if not self._no_deleted_spans:
                    # Find the deleted tokens, which appear after the current token.
                    deleted_tokens = []
                    for i in range(index + 1, source_length):
                        if i in source_token_ids_set:
                            break
                        deleted_tokens.append(source_token_ids[i])
                    # Bracket the deleted tokens, between unused0 and unused1.
                    if deleted_tokens:
                        deleted_tokens = (
                            [constants.DELETE_SPAN_START]
                            + list(
                                self._builder.tokenizer.convert_ids_to_tokens(
                                    deleted_tokens
                                )
                            )
                            + [constants.DELETE_SPAN_END]
                        )
                        out_tokens_with_deletes += deleted_tokens
            # Add the predicted phrase.
            elif "|" in tag:
                pos_pipe = tag.index("|")
                added_phrase = tag[pos_pipe + 1 :]
                out_tokens.append(added_phrase)

        if not self._use_open_vocab:
            out_tokens_with_deletes = out_tokens
        assert out_tokens_with_deletes[0] == (self._builder.tokenizer.cls_token), (
            f" {out_tokens_with_deletes} did not start/end with the correct tokens "
            f"{self._builder.tokenizer.cls_token}, {self._builder.tokenizer.sep_token}"
        )
        return out_tokens_with_deletes

    def _realize_tagging_wo_pointing_single(
        self, input_ids, last_token_index, tag_logits
    ):
        """Returns realized prediction for a given source for FelixInsert.

        TODO: Add special handling for [SEP] tokens like done above for the
        full Felix model.

        Args:
          input_ids:  Source token ids.
          last_token_index: The index, in the input_ids, of the last token (not
            including padding tokens).
          tag_logits: Tag logits  [vocab size, sequence_length].

        Returns:
          Realized predictions including deleted tokens.
        """

        input_tokens = self._builder.tokenizer.convert_ids_to_tokens(input_ids)
        predicted_tags = torch.argmax(tag_logits, dim=1).tolist()[
            : last_token_index + 1
        ]
        label_tuples = [self._inverse_label_map[int(tag)] for tag in predicted_tags]
        tokens = self._builder.build_insertion_tokens(input_tokens, label_tuples)
        if tokens is None:
            return None
        return tokens[0]


class JointPredictor(Predictor):
    """Predictor for joint models that output tagging and insertion predictions together."""

    def __init__(
        self,
        model_filepath: str,
        tokenizer_name: str,
        label_map_file: str,
        sequence_length: int = 128,
        use_open_vocab: bool = False,
        is_pointing: bool = False,
        special_glue_string_for_joining_sources: str = " ",
        use_token_type_ids: bool = False,
        no_deleted_spans: bool = False,
    ):
        """Initializes a JointPredictor instance.

        Args:
            model_filepath: Path to the joint model checkpoint.
            Other args: Same as Predictor.
        """
        super().__init__(
            model_tagging_filepath=None,
            model_insertion_filepath=None,
            tokenizer_name=tokenizer_name,
            label_map_file=label_map_file,
            sequence_length=sequence_length,
            use_open_vocab=use_open_vocab,
            is_pointing=is_pointing,
            special_glue_string_for_joining_sources=special_glue_string_for_joining_sources,
            use_token_type_ids=use_token_type_ids,
            no_deleted_spans=no_deleted_spans,
        )
        self._model_filepath = model_filepath
        self._joint_model = None

        if not is_pointing:
            raise ValueError("Pointing is not supported in joint models.")

    def _load_model(self):
        if self._joint_model is None:
            self._joint_model = AutoModel.from_pretrained(self._model_filepath)
            self._joint_model.eval()

    def _predict_batch(self, source_batch, is_insertion):
        self._load_model()
        if is_insertion:
            predictions = self._joint_model.get_insertion_predictions(**source_batch)
            return torch.argmax(predictions, dim=-1)

        if self._is_pointing:
            tag_logits, pointing_logits = self._joint_model.get_tagging_predictions(
                **source_batch
            )
            # Convert two lists into a single list of tuples.
            return list(zip(tag_logits, pointing_logits))
        else:
            tag_logits = self._tagging_model(**source_batch)
            return tag_logits

    def get_number_of_parameters(self):
        if self._joint_model is None:
            self._load_model()
        return sum(p.numel() for p in self._joint_model.parameters())
