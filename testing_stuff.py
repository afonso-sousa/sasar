import pointing_converter

input_texts = "A B C D".split()
target = "A B C D"
phrase_vocabulary = ["and"]
target_points = [1, 2, 3, 0]
target_phrase = ["", "", "", ""]

input_texts = "cls the big very loud cat sep".split()
target = "cls the very big old cat sep"
phrase_vocabulary = ["old"]
target_points = [1, 3, 5, 2, 0, 6, 0]
target_phrase = ["", "", "old", "", "", "", ""]

converter = pointing_converter.PointingConverter(phrase_vocabulary)
points = converter.compute_points(input_texts, target)
print([x.added_phrase for x in points])
print([x.point_index for x in points])

################
# input_texts = ["cls", "he", "was", "told", "a", "story", "sep"]
# target = "cls he tell a story sep"

################
# input_texts = [
#     "cls",
#     "The",
#     "quick",
#     "brown",
#     "fox",
#     "jumps",
#     "over",
#     "the",
#     "lazy",
#     "dog",
#     "sep",
# ]
# target = "cls A fast brown fox leaps above a dog that is lazy sep"
# target_points = ["a fast", "", "", "", "leaps above a", "", "", "", "", "that is", ""]
# target_phrase = [3, 0, 0, 4, 9, 0, 0, 0, 10, 8, 0]
################
# input_texts = [
#     "cls",
#     "The",
#     "novel",
#     "was",
#     "written",
#     "by",
#     "the",
#     "famous",
#     "author",
#     "in",
#     "just",
#     "three",
#     "months",
#     "sep",
# ]
# target = "cls The famous author wrote the novel in just three months sep"
# target_points = [
#     "the famous",
#     "",
#     "in",
#     "",
#     "the",
#     "",
#     "",
#     "",
#     "",
#     "",
#     "three months",
#     "",
#     "",
#     "",
# ]
# target_phrase = [8, 0, 10, 0, 2, 0, 0, 0, 4, 0, 13, 0, 0, 0]

# converter = pointing_converter.PointingConverter([], with_graph=True)
# points = converter.compute_points(input_texts, target)
# print([x.added_phrase for x in points])
# print([x.point_index for x in points])

################

# converter = pointing_converter.PointingConverter([], with_graph=True)
# closest_match = converter.find_closest_match(
#     "months",
#     [
#         "write-01",
#         ":ARG0",
#         "person",
#         ":ARG1-of",
#         "fame-01",
#         ":ARG0-of",
#         "author-01",
#         ":ARG1",
#         "novel",
#         ":duration",
#         "temporal-quantity",
#         ":quant",
#         "3",
#         ":unit",
#         "month",
#         ":mod",
#         "just",
#     ],
#     {
#         "cls": {0},
#         "the": {1, 6},
#         "novel": set(),
#         "was": {3},
#         "written": set(),
#         "by": {5},
#         "famous": {7},
#         "author": set(),
#         "in": {9},
#         "just": set(),
#         "three": {11},
#         "months": {12},
#         "sep": {13},
#         "wrote": set(),
#     },
# )
# print(closest_match)

# converter = pointing_converter.PointingConverter([], with_graph=True)
# closest_match = converter.find_closest_match(
#     "three",
#     [
#         "write-01",
#         ":ARG0",
#         "person",
#         ":ARG1-of",
#         "fame-01",
#         ":ARG0-of",
#         "author-01",
#         ":ARG1",
#         "novel",
#         ":duration",
#         "temporal-quantity",
#         ":quant",
#         "3",
#         ":unit",
#         "month",
#         ":mod",
#         "just",
#     ],
#     {
#         "cls": {0},
#         "the": {1, 6},
#         "novel": set(),
#         "was": {3},
#         "written": set(),
#         "by": {5},
#         "famous": {7},
#         "author": set(),
#         "in": {9},
#         "just": set(),
#         "three": {11},
#         "months": {12},
#         "sep": {13},
#         "wrote": set(),
#     },
# )
# print(closest_match)

# converter = pointing_converter.PointingConverter([], with_graph=True)
# closest_match = converter.find_closest_match(
#     "famous",
#     [
#         "write-01",
#         ":ARG0",
#         "person",
#         ":ARG1-of",
#         "fame-01",
#         ":ARG0-of",
#         "author-01",
#         ":ARG1",
#         "novel",
#         ":duration",
#         "temporal-quantity",
#         ":quant",
#         "3",
#         ":unit",
#         "month",
#         ":mod",
#         "just",
#     ],
#     {
#         "cls": {0},
#         "the": {1, 6},
#         "novel": set(),
#         "was": {3},
#         "written": set(),
#         "by": {5},
#         "famous": {7},
#         "author": set(),
#         "in": {9},
#         "just": set(),
#         "three": {11},
#         "months": {12},
#         "sep": {13},
#         "wrote": set(),
#     },
# )
# print(closest_match)

# import insertion_converter
# import transformer_example
# import utils

# max_seq_length = 128
# tokenizer_name = "bert-base-uncased"
# label_map_file = "input/label_map.json"

# label_map = utils.read_label_map(label_map_file, use_str_keys=True)

# converter_insertion = insertion_converter.InsertionConverter(
#     max_seq_length=max_seq_length,
#     label_map=label_map,
#     tokenizer_name=tokenizer_name,
# )
# converter_tagging = pointing_converter.PointingConverter(
#     {}, do_lower_case=True, with_graph=True
# )

# builder = transformer_example.TransformerExampleBuilder(
#     label_map=label_map,
#     tokenizer_name=tokenizer_name,
#     max_seq_length=max_seq_length,
#     converter=converter_tagging,
#     use_open_vocab=True,
#     converter_insertion=converter_insertion,
#     special_glue_string_for_sources=" ",
# )

# example, insertion_example = builder.build_transformer_example(
#     ["The novel was written by the famous author in just three months"],
#     "The famous author wrote the novel in just 3 months",
# )
# # example, insertion_example = builder.build_transformer_example(
# #     ["A simple sentence"],
# #     "that has nothing to do with the original",
# # )


# print(builder.tokenizer.convert_ids_to_tokens(insertion_example["input_ids"]))
# print(
#     builder.tokenizer.convert_ids_to_tokens(
#         [0 if a == -100 else a for a in insertion_example["masked_lm_ids"]]
#     )
# )
# breakpoint()

#########################

# import insertion_converter
# import transformer_example
# import utils

# max_seq_length = 128
# tokenizer_name = "bert-base-uncased"
# label_map_file = "input/label_map.json"

# label_map = utils.read_label_map(label_map_file, use_str_keys=True)

# converter_insertion = insertion_converter.InsertionConverter(
#     max_seq_length=max_seq_length,
#     label_map=label_map,
#     tokenizer_name=tokenizer_name,
# )
# converter_tagging = pointing_converter.PointingConverter(
#     {}, do_lower_case=True, with_graph=True
# )

# builder = transformer_example.TransformerExampleBuilder(
#     label_map=label_map,
#     tokenizer_name=tokenizer_name,
#     max_seq_length=max_seq_length,
#     converter=converter_tagging,
#     use_open_vocab=True,
#     converter_insertion=converter_insertion,
#     special_glue_string_for_sources=" ",
# )

# amr_source = '(c / conflict-01\n      :ARG0 (p / province\n            :name (n / name\n                  :op1 "Baluchistan"))\n      :ARG2 (a / amr-unknown\n            :mod (e / exact)))'
# amr_target = '(m / multi-sentence\n      :snt1 (c / conflict-01\n            :ARG0 (p / province\n                  :name (n / name\n                        :op1 "Baluchistan"))\n            :ARG2 (a / amr-unknown))\n      :snt2 (w / want-01\n            :ARG0 (p2 / person\n                  :mod p)\n            :ARG1 (a2 / amr-unknown)))'


# example, insertion_example = builder.build_transformer_example(
#     ["What exactly is the Baluchistan conflict?"],
#     "What is the Baluchistan conflict? What do the people of Baluchistan want?",
#     amr_source,
#     amr_target,
# )

# print(
#     builder.tokenizer.convert_ids_to_tokens(
#         [101, 2054, 3599, 2003, 1996, 28352, 15217, 12693, 4736, 1029, 102]
#     )
# )
# print(example.point_indexes)
# print(builder.tokenizer.convert_ids_to_tokens(insertion_example["input_ids"]))
# print(
#     builder.tokenizer.convert_ids_to_tokens(
#         [0 if a == -100 else a for a in insertion_example["masked_lm_ids"]]
#     )
# )

#################
import insertion_converter
import transformer_example
import utils

max_seq_length = 128
tokenizer_name = "bert-base-uncased"
label_map_file = "input/label_map.json"

label_map = utils.read_label_map(label_map_file, use_str_keys=True)

converter_insertion = insertion_converter.InsertionConverter(
    max_seq_length=max_seq_length,
    label_map=label_map,
    tokenizer_name=tokenizer_name,
)
converter_tagging = pointing_converter.PointingConverter(
    {}, do_lower_case=True, with_graph=True
)

builder = transformer_example.TransformerExampleBuilder(
    label_map=label_map,
    tokenizer_name=tokenizer_name,
    max_seq_length=max_seq_length,
    converter=converter_tagging,
    use_open_vocab=True,
    converter_insertion=converter_insertion,
    special_glue_string_for_sources=" ",
)

amr_source = "(w / write-01\n   :ARG0 (a / author\n :mod (f / famous))\n    :ARG1 (n / novel)\n :time (m / month\n  :quant 3\n  :mod (j / just)))"
amr_target = "(w / write-01\n   :ARG0 (a / author\n :mod (f / famous))\n    :ARG1 (n / novel)\n :time (m / month\n  :quant 3\n  :mod (j / just)))"

example, insertion_example = builder.build_transformer_example(
    ["The novel was written by the famous author in just three months"],
    "The famous author wrote the novel in just three months",
    amr_source,
    amr_target,
)

print(
    builder.tokenizer.convert_ids_to_tokens(
        [101, 2054, 3599, 2003, 1996, 28352, 15217, 12693, 4736, 1029, 102]
    )
)
print(example.point_indexes)
print(builder.tokenizer.convert_ids_to_tokens(insertion_example["input_ids"]))
print(
    builder.tokenizer.convert_ids_to_tokens(
        [0 if a == -100 else a for a in insertion_example["masked_lm_ids"]]
    )
)
