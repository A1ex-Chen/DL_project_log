def find_pipeline_class(loaded_module):
    """
    Retrieve pipeline class that inherits from `DiffusionPipeline`. Note that there has to be exactly one class
    inheriting from `DiffusionPipeline`.
    """
    from ..pipelines import DiffusionPipeline
    cls_members = dict(inspect.getmembers(loaded_module, inspect.isclass))
    pipeline_class = None
    for cls_name, cls in cls_members.items():
        if cls_name != DiffusionPipeline.__name__ and issubclass(cls,
            DiffusionPipeline) and cls.__module__.split('.')[0] != 'diffusers':
            if pipeline_class is not None:
                raise ValueError(
                    f'Multiple classes that inherit from {DiffusionPipeline.__name__} have been found: {pipeline_class.__name__}, and {cls_name}. Please make sure to define only one in {loaded_module}.'
                    )
            pipeline_class = cls
    return pipeline_class
