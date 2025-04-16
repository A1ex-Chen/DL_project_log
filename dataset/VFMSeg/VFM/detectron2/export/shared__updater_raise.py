def _updater_raise(op, input_types, output_types):
    raise RuntimeError(
        'Failed to apply updater for op {} given input_types {} and output_types {}'
        .format(op, input_types, output_types))
