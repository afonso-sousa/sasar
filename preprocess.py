import bert_example
import insertion_converter
import pointing_converter
import utils


def initialize_builder(
    use_open_vocab,
    label_map_file,
    max_seq_length,
    tokenizer_name,
    special_glue_string_for_sources,
):
    """Returns a builder for tagging and insertion BERT examples."""
    label_map = utils.read_label_map(label_map_file, use_str_keys=True)

    if use_open_vocab:
        converter_insertion = insertion_converter.InsertionConverter(
            max_seq_length=max_seq_length,
            label_map=label_map,
            tokenizer_name=tokenizer_name,
        )
        converter_tagging = pointing_converter.PointingConverter({}, do_lower_case=True)

    builder = bert_example.BertExampleBuilder(
        label_map=label_map,
        tokenizer_name=tokenizer_name,
        max_seq_length=max_seq_length,
        converter=converter_tagging,
        use_open_vocab=use_open_vocab,
        converter_insertion=converter_insertion,
        special_glue_string_for_sources=special_glue_string_for_sources,
    )

    return builder
