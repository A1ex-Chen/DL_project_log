def input_provider_wrapped(batch_size=None):
    if batch_size is None:
        inputs = input_provider()
    else:
        inputs = input_provider(batch_size=batch_size)
    if isinstance(inputs, torch.Tensor):
        raise AnalysisError(
            "The input provider function must return an iterable that contains the inputs for the model. If your model only takes a single tensor as input, return a single element tuple or list in your input provider (e.g., 'return [the_input]')."
            ).with_file_context(entry_point)
    try:
        iter(inputs)
        return inputs
    except TypeError:
        raise AnalysisError(
            'The input provider function must return an iterable that contains the inputs for the model.'
            ).with_file_context(entry_point)
