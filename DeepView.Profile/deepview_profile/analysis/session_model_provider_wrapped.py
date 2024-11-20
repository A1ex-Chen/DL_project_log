def model_provider_wrapped():
    model = model_provider()
    if not callable(model):
        raise AnalysisError(
            'The model provider function must return a callable (i.e. return something that can be called like a PyTorch module or function).'
            ).with_file_context(entry_point)
    return model
