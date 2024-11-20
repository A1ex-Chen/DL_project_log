def _get_custom_pipeline_class(custom_pipeline, repo_id=None, hub_revision=
    None, class_name=None, cache_dir=None, revision=None):
    if custom_pipeline.endswith('.py'):
        path = Path(custom_pipeline)
        file_name = path.name
        custom_pipeline = path.parent.absolute()
    elif repo_id is not None:
        file_name = f'{custom_pipeline}.py'
        custom_pipeline = repo_id
    else:
        file_name = CUSTOM_PIPELINE_FILE_NAME
    if repo_id is not None and hub_revision is not None:
        revision = hub_revision
    return get_class_from_dynamic_module(custom_pipeline, module_file=
        file_name, class_name=class_name, cache_dir=cache_dir, revision=
        revision)
