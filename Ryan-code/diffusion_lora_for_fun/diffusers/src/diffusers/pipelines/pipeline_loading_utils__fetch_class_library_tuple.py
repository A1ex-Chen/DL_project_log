def _fetch_class_library_tuple(module):
    diffusers_module = importlib.import_module(__name__.split('.')[0])
    pipelines = getattr(diffusers_module, 'pipelines')
    not_compiled_module = _unwrap_model(module)
    library = not_compiled_module.__module__.split('.')[0]
    module_path_items = not_compiled_module.__module__.split('.')
    pipeline_dir = module_path_items[-2] if len(module_path_items
        ) > 2 else None
    path = not_compiled_module.__module__.split('.')
    is_pipeline_module = pipeline_dir in path and hasattr(pipelines,
        pipeline_dir)
    if is_pipeline_module:
        library = pipeline_dir
    elif library not in LOADABLE_CLASSES:
        library = not_compiled_module.__module__
    class_name = not_compiled_module.__class__.__name__
    return library, class_name
