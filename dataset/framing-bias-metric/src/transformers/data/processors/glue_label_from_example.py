def label_from_example(example: InputExample) ->Union[int, float, None]:
    if example.label is None:
        return None
    if output_mode == 'classification':
        return label_map[example.label]
    elif output_mode == 'regression':
        return float(example.label)
    raise KeyError(output_mode)
