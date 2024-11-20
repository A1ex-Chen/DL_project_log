def _other_ops_updater(op, input_types, output_types):
    non_none_types = [x for x in input_types + output_types if x is not None]
    if len(non_none_types) > 0:
        the_type = non_none_types[0]
        if not all(x == the_type for x in non_none_types):
            _updater_raise(op, input_types, output_types)
    else:
        the_type = None
    return [the_type for _ in op.input], [the_type for _ in op.output]
