def _get_pipeline_class(class_obj, config=None, load_connected_pipeline=
    False, custom_pipeline=None, repo_id=None, hub_revision=None,
    class_name=None, cache_dir=None, revision=None):
    if custom_pipeline is not None:
        return _get_custom_pipeline_class(custom_pipeline, repo_id=repo_id,
            hub_revision=hub_revision, class_name=class_name, cache_dir=
            cache_dir, revision=revision)
    if class_obj.__name__ != 'DiffusionPipeline':
        return class_obj
    diffusers_module = importlib.import_module(class_obj.__module__.split(
        '.')[0])
    class_name = class_name or config['_class_name']
    if not class_name:
        raise ValueError(
            'The class name could not be found in the configuration file. Please make sure to pass the correct `class_name`.'
            )
    class_name = class_name[4:] if class_name.startswith('Flax'
        ) else class_name
    pipeline_cls = getattr(diffusers_module, class_name)
    if load_connected_pipeline:
        from .auto_pipeline import _get_connected_pipeline
        connected_pipeline_cls = _get_connected_pipeline(pipeline_cls)
        if connected_pipeline_cls is not None:
            logger.info(
                f'Loading connected pipeline {connected_pipeline_cls.__name__} instead of {pipeline_cls.__name__} as specified via `load_connected_pipeline=True`'
                )
        else:
            logger.info(
                f'{pipeline_cls.__name__} has no connected pipeline class. Loading {pipeline_cls.__name__}.'
                )
        pipeline_cls = connected_pipeline_cls or pipeline_cls
    return pipeline_cls
