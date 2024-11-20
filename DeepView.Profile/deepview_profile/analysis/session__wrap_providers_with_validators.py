def _wrap_providers_with_validators(model_provider, input_provider,
    iteration_provider, entry_point):

    def model_provider_wrapped():
        model = model_provider()
        if not callable(model):
            raise AnalysisError(
                'The model provider function must return a callable (i.e. return something that can be called like a PyTorch module or function).'
                ).with_file_context(entry_point)
        return model

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

    def iteration_provider_wrapped(model):
        iteration = iteration_provider(model)
        if not callable(iteration):
            raise AnalysisError(
                'The iteration provider function must return a callable (i.e. return something that can be called like a function).'
                ).with_file_context(entry_point)
        return iteration
    return (model_provider_wrapped, input_provider_wrapped,
        iteration_provider_wrapped)
