def _validate_providers_signatures(model_provider, input_provider,
    iteration_provider, entry_point):
    model_sig = inspect.signature(model_provider)
    if len(model_sig.parameters) != 0:
        raise AnalysisError(
            'The model provider function cannot have any parameters.'
            ).with_file_context(entry_point)
    input_sig = inspect.signature(input_provider)
    if len(input_sig.parameters
        ) != 1 or BATCH_SIZE_ARG not in input_sig.parameters or type(input_sig
        .parameters[BATCH_SIZE_ARG].default) is not int:
        raise AnalysisError(
            "The input provider function must have exactly one '{}' parameter with an integral default value."
            .format(BATCH_SIZE_ARG)).with_file_context(entry_point)
    batch_size = input_sig.parameters[BATCH_SIZE_ARG].default
    iteration_sig = inspect.signature(iteration_provider)
    if len(iteration_sig.parameters) != 1:
        raise AnalysisError(
            'The iteration provider function must have exactly one parameter (the model being profiled).'
            ).with_file_context(entry_point)
    return batch_size
