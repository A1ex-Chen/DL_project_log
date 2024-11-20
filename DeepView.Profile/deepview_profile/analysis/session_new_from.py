@classmethod
def new_from(cls, project_root, entry_point):
    path_to_entry_point = os.path.join(project_root, entry_point)
    path_to_entry_point_dir = os.path.dirname(path_to_entry_point)
    entry_point_code, entry_point_ast, scope = _run_entry_point(
        path_to_entry_point, path_to_entry_point_dir, project_root)
    if MODEL_PROVIDER_NAME not in scope:
        raise AnalysisError(
            'The project entry point file is missing a model provider function. Please add a model provider function named "{}".'
            .format(MODEL_PROVIDER_NAME)).with_file_context(entry_point)
    if INPUT_PROVIDER_NAME not in scope:
        raise AnalysisError(
            'The project entry point file is missing an input provider function. Please add an input provider function named "{}".'
            .format(INPUT_PROVIDER_NAME)).with_file_context(entry_point)
    if ITERATION_PROVIDER_NAME not in scope:
        raise AnalysisError(
            'The project entry point file is missing an iteration provider function. Please add an iteration provider function named "{}".'
            .format(ITERATION_PROVIDER_NAME)).with_file_context(entry_point)
    batch_size = _validate_providers_signatures(scope[MODEL_PROVIDER_NAME],
        scope[INPUT_PROVIDER_NAME], scope[ITERATION_PROVIDER_NAME], entry_point
        )
    model_provider, input_provider, iteration_provider = (
        _wrap_providers_with_validators(scope[MODEL_PROVIDER_NAME], scope[
        INPUT_PROVIDER_NAME], scope[ITERATION_PROVIDER_NAME], entry_point))
    return cls(project_root, entry_point, path_to_entry_point_dir,
        model_provider, input_provider, iteration_provider, batch_size,
        StaticAnalyzer(entry_point_code, entry_point_ast))
