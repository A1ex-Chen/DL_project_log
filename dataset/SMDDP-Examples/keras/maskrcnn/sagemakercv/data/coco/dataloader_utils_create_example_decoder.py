def create_example_decoder():
    return TfExampleDecoder(use_instance_mask=use_instance_mask,
        regenerate_source_id=regenerate_source_id)
