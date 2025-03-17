import insertion_converter
import pointing_converter
import transformer_example
import utils


def initialize_builder(
    use_open_vocab,
    label_map_file,
    max_seq_length,
    tokenizer_name,
    special_glue_string_for_sources,
    with_graph=None,
    include_deleted_spans=True,
):
    """Returns a builder for tagging and insertion Transformer examples."""
    label_map = utils.read_label_map(label_map_file, use_str_keys=True)

    if use_open_vocab:
        converter_insertion = insertion_converter.InsertionConverter(
            max_seq_length=max_seq_length,
            label_map=label_map,
            tokenizer_name=tokenizer_name,
            include_deleted_spans=include_deleted_spans,
        )
        converter_tagging = pointing_converter.PointingConverter(
            {},
            do_lower_case=True,
            with_graph=with_graph,
        )
    else:
        raise ValueError("Open vocabulary is required for this task.")

    builder = transformer_example.TransformerExampleBuilder(
        label_map=label_map,
        tokenizer_name=tokenizer_name,
        max_seq_length=max_seq_length,
        converter=converter_tagging,
        use_open_vocab=use_open_vocab,
        converter_insertion=converter_insertion,
        special_glue_string_for_sources=special_glue_string_for_sources,
    )

    return builder
