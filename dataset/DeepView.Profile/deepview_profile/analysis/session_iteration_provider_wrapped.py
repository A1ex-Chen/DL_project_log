def iteration_provider_wrapped(model):
    iteration = iteration_provider(model)
    if not callable(iteration):
        raise AnalysisError(
            'The iteration provider function must return a callable (i.e. return something that can be called like a function).'
            ).with_file_context(entry_point)
    return iteration
