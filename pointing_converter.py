import collections

import pointing


class PointingConverter(object):
    """Converter from training target texts into pointing format."""

    def __init__(self, phrase_vocabulary, do_lower_case=True, with_graph=False):
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

    def compute_points(self, source_tokens, target):
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
