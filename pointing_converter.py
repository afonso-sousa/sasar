import collections
import json
import os
import string

import amrlib
import penman
from nltk.corpus import wordnet
from word2number import w2n

import amr_utils
import pointing


def detokenize(tokens):
    detok = []
    for token in tokens:
        if token.startswith("##"):  ## Only for BERT tokenization
            detok[-1] += token[2:]  # Merge subword with previous token
        elif token in string.punctuation or token.startswith("'"):
            detok.append(token)  # Attach punctuation without space
        else:
            detok.append(" " + token)  # Add space before normal words
    return "".join(detok).strip()


class PointingConverter(object):
    """Converter from training target texts into pointing format."""

    def __init__(
        self,
        phrase_vocabulary,
        do_lower_case=True,
        with_graph=False,
    ):
        """Initializes an instance of PointingConverter.

        Args:
          phrase_vocabulary: Iterable of phrase vocabulary items (strings), if empty
            we assume an unlimited vocabulary.
          do_lower_case: Should the phrase vocabulary be lower cased.
        """
        self._with_graph = with_graph
        self._do_lower_case = do_lower_case
        self._phrase_vocabulary = set()
        for phrase in phrase_vocabulary:
            if do_lower_case:
                phrase = phrase.lower()
            # Remove the KEEP/DELETE flags for vocabulary phrases.
            if "|" in phrase:
                self._phrase_vocabulary.add(phrase.split("|")[1])
            else:
                self._phrase_vocabulary.add(phrase)

        if self._with_graph:
            from nltk.stem import WordNetLemmatizer

            self.lemmatizer = WordNetLemmatizer()

            self.stog = amrlib.load_stog_model(
                model_dir="amr_parser",
                device="cuda:0",
                batch_size=2,
            )

    def compute_points(
        self,
        source_tokens,
        target,
        amr_source=None,
        amr_target=None,
        source_word_ids=None,
        target_word_ids=None,
    ):
        """Computes points needed for converting the source into the target.

        Args:
          source_tokens: Source tokens.
          target: Target text.

        Returns:
          List of pointing.Point objects. If the source couldn't be converted into the target via pointing, returns an empty list.
        """
        if self._do_lower_case:
            target = target.lower()
            source_tokens = [x.lower() for x in source_tokens]
        target_tokens = target.split()

        if self._with_graph:
            source_text = " ".join(source_tokens[1:-1])
            if not amr_source or not amr_target:
                # Generate AMR graphs for source and target texts
                source_text = detokenize(source_tokens[1:-1])  # Exclude special tokens
                target_text = detokenize(target_tokens[1:-1])
                amr_graphs = self.stog.parse_sents([source_text, target_text])
                amr_source = amr_graphs[0].split("\n", 1)[1]
                amr_target = amr_graphs[1].split("\n", 1)[1]

            points = self._compute_points_from_AMR(
                source_tokens,
                target_tokens,
                amr_source,
                amr_target,
                source_word_ids,
                target_word_ids,
            )
        else:
            points = self._compute_points(source_tokens, target_tokens)
        return points

    def _compute_points(self, source_tokens, target_tokens):
        """Computes points needed for converting the source into the target.

        Args:
          source_tokens: List of source tokens.
          target_tokens: List of target tokens.

        Returns:
          List of pointing.Pointing objects. If the source couldn't be converted into the target via pointing, returns an empty list.
        """
        source_tokens_indexes = collections.defaultdict(set)
        for i, source_token in enumerate(source_tokens):
            source_tokens_indexes[source_token].add(i)

        target_points = {}
        last = 0
        token_buffer = ""

        def find_nearest(indexes, index):
            # In the case that two indexes are equally far apart
            # the lowest index is returned.
            return min(indexes, key=lambda x: abs(x - index))

        for target_token in target_tokens[1:]:
            # Is the target token in the source tokens and is buffer in the vocabulary " ##" converts word pieces into words
            if source_tokens_indexes[target_token] and (
                not token_buffer
                or not self._phrase_vocabulary
                or token_buffer in self._phrase_vocabulary
            ):
                # Maximum length expected of source_tokens_indexes[target_token] is 512,
                # median length is 1.
                src_indx = find_nearest(source_tokens_indexes[target_token], last)
                # We can only point to a token once.
                source_tokens_indexes[target_token].remove(src_indx)
                target_points[last] = pointing.Point(src_indx, token_buffer)
                last = src_indx
                token_buffer = ""

            else:
                token_buffer = (token_buffer + " " + target_token).strip()

        ## Buffer needs to be empty at the end.
        if token_buffer.strip():
            return []

        points = []
        for i in range(len(source_tokens)):
            ## If a source token is not pointed to,
            ## then it should point to the start of the sequence.
            if i not in target_points:
                points.append(pointing.Point(0))
            else:
                points.append(target_points[i])

        return points

    def reconstruct_word_from_tokens(self, tokens, word_ids, target_word_id):
        return "".join(
            token[2:] if token.startswith("##") else token
            for token, wid in zip(tokens, word_ids)
            if wid == target_word_id
        )

    def _compute_points_from_AMR(
        self,
        source_tokens,
        target_tokens,
        amr_source,
        amr_target,
        source_word_ids,
        target_word_ids,
    ):
        # Extract important concepts and relations
        source_amr_tokens = self.extract_amr_tokens(amr_source)
        target_amr_tokens = self.extract_amr_tokens(amr_target)

        # Find shared concepts
        common_concepts = set(source_amr_tokens) & set(target_amr_tokens)
        common_concepts = {
            c.lower() if self._do_lower_case else c
            for c in common_concepts
            if not c.startswith(":")
        }
        common_concepts.add(source_tokens[-1])  # add eos token

        # Create a map of source tokens to indices
        source_words_indexes = collections.defaultdict(list)
        breakpoint()
        for i, word_id in enumerate(source_word_ids):
            if word_id is not None:
                word = self.reconstruct_word_from_tokens(
                    source_tokens, source_word_ids, word_id
                )
                # Check if this is a continuation of the previous word
                if (
                    source_words_indexes.get(word)
                    and i == source_words_indexes[word][-1][-1] + 1
                    and source_tokens[i].startswith("##")
                ):
                    # Append to the last occurrence's indices
                    source_words_indexes[word][-1].append(i)
                else:
                    # Start a new occurrence
                    source_words_indexes[word].append([i])

        source_words_indexes["[sep]"].append([i])

        # Initialize pointing mechanism
        target_points = {}
        last = 0
        token_buffer = ""

        def find_nearest(index_groups, index):
            # In the case that two indexes are equally far apart
            # the lowest first element index is returned.
            return min(index_groups, key=lambda group: abs(group[0] - index))

        target_words = []
        current_word_id = None
        for i, word_id in enumerate(target_word_ids[1:]):  # skip first token
            if word_id != current_word_id:
                # New word found
                word = self.reconstruct_word_from_tokens(
                    target_tokens[1:], target_word_ids[1:], word_id
                )
                target_words.append(word)
                current_word_id = word_id

        for target_word in target_words:
            if (
                source_words_indexes.get(target_word) and target_word in common_concepts
            ) and (
                not token_buffer
                or not self._phrase_vocabulary
                or token_buffer in self._phrase_vocabulary
            ):
                src_indx = find_nearest(source_words_indexes[target_word], last)
                # We can only point to a token once.
                source_words_indexes[target_word].remove(src_indx)
                for idx in src_indx:
                    target_points[last] = pointing.Point(idx, token_buffer)
                    token_buffer = ""
                    last = idx
            elif aligned_source_token := self.find_closest_match(
                target_word, common_concepts, source_words_indexes
            ):
                if not source_words_indexes.get(aligned_source_token):
                    breakpoint()
                src_indx = find_nearest(
                    source_words_indexes[aligned_source_token], last
                )
                source_words_indexes[aligned_source_token].remove(src_indx)
                for idx in src_indx:
                    target_points[last] = pointing.Point(idx, token_buffer)
                    token_buffer = ""
                    last = idx
            else:
                token_buffer = (token_buffer + " " + target_word).strip()

        # Ensure buffer is empty
        if token_buffer.strip():
            return []

        # Construct the final pointer map
        points = []
        for i in range(len(source_tokens)):
            ## If a source token is not pointed to,
            ## then it should point to the start of the sequence.
            if i not in target_points:
                points.append(pointing.Point(0))
            else:
                points.append(target_points[i])

        return points

    def extract_amr_tokens(self, amr_str):
        """
        Extract key concepts from an AMR graph.
        Returns a set of relevant concept names.
        """
        graph_penman = penman.decode(amr_str)
        v2c_penman = amr_utils.map_variables_to_concepts(graph_penman)

        # Linearize and clean up the AMR
        linearized_amr = penman.encode(graph_penman).replace("\t", "").replace("\n", "")
        tokens = linearized_amr.split()

        # Simplify the tokens
        new_tokens, _ = amr_utils.simplify_amr_tokens(tokens, v2c_penman)

        return new_tokens

    def find_closest_match(
        self, target_token, common_amr_concepts, source_tokens_indexes
    ):
        def get_wordnet_pos(word):
            tag = (
                wordnet.synsets(word)[0].pos()
                if wordnet.synsets(word)
                else wordnet.NOUN
            )
            tag = tag[0].upper()
            tag_dict = {
                "J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV,
            }

            return tag_dict.get(tag, wordnet.NOUN)

        def convert_to_number(token):
            try:
                return str(w2n.word_to_num(token))
            except ValueError:
                return token

        non_empty_source_tokens = [
            key for key, value in source_tokens_indexes.items() if value
        ]

        if (
            target_token == "?"
            and "?" in non_empty_source_tokens
            and "amr-unknown" in common_amr_concepts
        ):
            return "?"

        target_token_lemma = self.lemmatizer.lemmatize(
            target_token, get_wordnet_pos(target_token)
        )
        source_amr_lemmas = [
            token.split("-")[0]
            for token in common_amr_concepts
            # if re.match(r"^[a-zA-Z]+-\d{2}$", token)
            if not token.startswith(":")
        ]

        # Try converting the target token into a number (e.g. "three" -> 3)
        target_token_lemma = convert_to_number(target_token_lemma)

        if target_token_lemma in source_amr_lemmas:
            for word in non_empty_source_tokens:
                processed_word = convert_to_number(word)

                if target_token_lemma == self.lemmatizer.lemmatize(
                    processed_word, get_wordnet_pos(processed_word)
                ):
                    return word

        return None
