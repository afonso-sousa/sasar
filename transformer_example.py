import collections

from transformers import AutoTokenizer


class TransformerExample:
    """Class for training and inference examples for Transformers.

    Attributes:
        features: A dictionary of features with tensor lists as values.
        features_float: A dictionary of features with float tensor lists as values.
        scalar_features: A dictionary of features with scalar values.
    """

    def __init__(
        self,
        input_ids,
        input_mask,
        token_type_ids,
        labels,
        point_indexes,
        labels_mask,
    ):
        """Constructor for TransformerExample.

        Args:
            input_ids: Tensor of ids of source tokens.
            input_mask: Tensor of 1s and 0s. 0 indicates a PAD token.
            token_type_ids: Tensor of segment ids.
            labels: Tensor of added phrases. If tensor is empty, we assume we are at test time.
            point_indexes: Tensor of target points.
            labels_mask: Tensor of 1s and 0s. 0 indicates a PAD token.
        """
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        self.point_indexes = point_indexes if labels else None
        self.labels = labels if labels else None
        self.labels_mask = labels_mask if labels else None

    def to_dict(self):
        # Get the dictionary representation of the object's attributes
        obj_dict = self.__dict__.copy()

        # Since some attributes might be None, we remove them from the
        # dictionary to avoid storing unnecessary information.
        attributes = [
            "input_ids",
            "point_indexes",
            "input_mask",
            "token_type_ids",
            "labels",
            "labels_mask",
        ]
        for feature_name in attributes:
            if obj_dict[feature_name] is None:
                obj_dict.pop(feature_name)

        return obj_dict

    def __repr__(self):
        return f"TransformerExample({self.to_dict()})"


class TransformerExampleBuilder:
    """Builder class for TransformerExample objects.

    Attributes:
      label_map: Mapping from tags to tag IDs.
      tokenizer: A tokenization.FullTokenizer, which converts between strings and
        lists of tokens.
    """

    def __init__(
        self,
        label_map,
        max_seq_length,
        converter,
        use_open_vocab,
        tokenizer_name="bert-base-uncased",
        converter_insertion=None,
        special_glue_string_for_sources=None,
    ):
        """Initializes an instance of TransformerExampleBuilder.

        Args:
          label_map: Mapping from tags to tag IDs.
          max_seq_length: Maximum sequence length.
          do_lower_case: Whether to lower case the input text. Should be True for
            uncased models and False for cased models.
          converter: Converter from text targets to points.
          use_open_vocab: Should MASK be inserted or phrases. Currently only True is supported.
          converter_insertion: Converter for building an insertion example based on
            the tagger output. Optional.
          special_glue_string_for_sources: If there are multiple sources, this
            string is used to combine them into one string. The empty string is a valid value. Optional.
        """
        self.label_map = label_map
        inverse_label_map = {}
        for label, label_id in label_map.items():
            if label_id in inverse_label_map:
                raise ValueError(
                    "Multiple labels with the same ID: {}".format(label_id)
                )
            inverse_label_map[label_id] = label
        self._inverse_label_map = inverse_label_map
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
        )
        self._max_seq_length = max_seq_length
        self._converter = converter
        self._pad_id = self.tokenizer.pad_token_id
        self._do_lower_case = (
            self.tokenizer.do_lower_case
            if hasattr(self.tokenizer, "do_lower_case")
            else True
        )
        self._use_open_vocab = use_open_vocab
        self._converter_insertion = converter_insertion
        if special_glue_string_for_sources is not None:
            self._special_glue_string_for_sources = special_glue_string_for_sources
        else:
            self._special_glue_string_for_sources = " "

    def build_transformer_example(
        self,
        sources,
        target=None,
        amr_source=None,
        amr_target=None,
        is_test_time=False,
        use_token_type_ids=False,
    ):
        """Constructs a Transformer tagging and insertion examples.

        Args:
          sources: List of source texts.
          target: Target text or None when building an example during inference. If the target is None then we don't calculate gold labels or tags, this is equivalent to setting is_test_time to True.
          is_test_time: Controls whether the dataset is to be used at test time. Unlike setting target = None to indicate test time, this flags allows for saving the target in the tfrecord.

        Returns:
          A tuple with:
          1. TransformerExample for the tagging model or None if there's a tag not found in self.label_map or conversion from text to tags was infeasible.
          2. FeedDict for the insertion model or None if the TransformerExample or the insertion conversion failed.
        """
        merged_sources = self._special_glue_string_for_sources.join(sources)
        merged_sources = merged_sources.strip()
        if self._do_lower_case:
            merged_sources = merged_sources.lower()
            # Ensure [SEP] or equivalent separator token is always uppercase.
            if self.tokenizer.sep_token:
                merged_sources = merged_sources.replace(
                    self.tokenizer.sep_token.lower(), self.tokenizer.sep_token
                )

        tokenized_input = self.tokenizer(
            merged_sources,
            max_length=self._max_seq_length,
            padding=False,
            truncation=True,
            return_token_type_ids=use_token_type_ids,
        )
        source_word_ids = tokenized_input.word_ids()

        input_ids = tokenized_input.input_ids
        input_mask = tokenized_input.attention_mask
        token_type_ids = getattr(tokenized_input, "token_type_ids", None)

        if not target or is_test_time:
            example = TransformerExample(
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=token_type_ids,
                labels=[],
                point_indexes=[],
                labels_mask=[],
            )
            return example, None

        if self._do_lower_case:
            target = target.lower()

        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        tokenized_output = self.tokenizer(
            target,
            max_length=self._max_seq_length,
            padding=False,
            truncation=True,
        )
        target_word_ids = tokenized_output.word_ids()
        output_tokens = self.tokenizer.convert_ids_to_tokens(tokenized_output.input_ids)
        points = self._converter.compute_points(
            " ".join(input_tokens).split(),
            " ".join(output_tokens),
            amr_source,
            amr_target,
            source_word_ids,
            target_word_ids,
        )

        if not points:
            return None, None

        labels = [t.added_phrase for t in points]

        point_indexes = [t.point_index for t in points]
        point_indexes_set = set(point_indexes)
        try:
            new_labels = []
            for i, added_phrase in enumerate(labels):
                if i not in point_indexes_set:
                    new_labels.append(self.label_map["DELETE"])
                elif not added_phrase:  # added phrase is ''
                    new_labels.append(self.label_map["KEEP"])
                else:
                    if self._use_open_vocab:
                        new_labels.append(
                            self.label_map["KEEP|" + str(len(added_phrase.split()))]
                        )
                    else:
                        new_labels.append(self.label_map["KEEP|" + str(added_phrase)])
                labels = new_labels
        except KeyError:
            print(f"Added_phrase ({added_phrase}) is not in label_map.")
            return None, None

        if not labels:
            return None, None

        label_counter = collections.Counter(labels)
        label_weight = {
            label: len(labels) / count / len(label_counter)
            for label, count in label_counter.items()
        }

        # Weight the labels inversely proportional to their frequency.
        labels_mask = [label_weight[label] for label in labels]
        example = TransformerExample(
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            point_indexes=point_indexes,
            labels_mask=labels_mask,
        )

        insertion_example = None
        if self._converter_insertion is not None:
            insertion_example = self._converter_insertion.create_insertion_example(
                input_tokens, labels, point_indexes, output_tokens
            )

        return example, insertion_example
