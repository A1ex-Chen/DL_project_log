def infer_shapes(nlp: Pipeline, framework: str) ->Tuple[List[str], List[str
    ], Dict, BatchEncoding]:
    """
    Attempt to infer the static vs dynamic axes for each input and output tensors for a specific model

    Args:
        nlp: The pipeline object holding the model to be exported
        framework: The framework identifier to dispatch to the correct inference scheme (pt/tf)

    Returns:

        - List of the inferred input variable names
        - List of the inferred output variable names
        - Dictionary with input/output variables names as key and shape tensor as value
        - a BatchEncoding reference which was used to infer all the above information
    """

    def build_shape_dict(name: str, tensor, is_input: bool, seq_len: int):
        if isinstance(tensor, (tuple, list)):
            return [build_shape_dict(name, t, is_input, seq_len) for t in
                tensor]
        else:
            axes = {[axis for axis, numel in enumerate(tensor.shape) if 
                numel == 1][0]: 'batch'}
            if is_input:
                if len(tensor.shape) == 2:
                    axes[1] = 'sequence'
                else:
                    raise ValueError(
                        f'Unable to infer tensor axes ({len(tensor.shape)})')
            else:
                seq_axes = [dim for dim, shape in enumerate(tensor.shape) if
                    shape == seq_len]
                axes.update({dim: 'sequence' for dim in seq_axes})
        print(
            f"Found {'input' if is_input else 'output'} {name} with shape: {axes}"
            )
        return axes
    tokens = nlp.tokenizer('This is a sample output', return_tensors=framework)
    seq_len = tokens.input_ids.shape[-1]
    outputs = nlp.model(**tokens) if framework == 'pt' else nlp.model(tokens)
    if isinstance(outputs, ModelOutput):
        outputs = outputs.to_tuple()
    if not isinstance(outputs, (list, tuple)):
        outputs = outputs,
    input_vars = list(tokens.keys())
    input_dynamic_axes = {k: build_shape_dict(k, v, True, seq_len) for k, v in
        tokens.items()}
    outputs_flat = []
    for output in outputs:
        if isinstance(output, (tuple, list)):
            outputs_flat.extend(output)
        else:
            outputs_flat.append(output)
    output_names = [f'output_{i}' for i in range(len(outputs_flat))]
    output_dynamic_axes = {k: build_shape_dict(k, v, False, seq_len) for k,
        v in zip(output_names, outputs_flat)}
    dynamic_axes = dict(input_dynamic_axes, **output_dynamic_axes)
    return input_vars, output_names, dynamic_axes, tokens
