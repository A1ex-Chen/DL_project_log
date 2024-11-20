def register_modules(self, **kwargs):
    from diffusers import pipelines
    for name, module in kwargs.items():
        if module is None:
            register_dict = {name: (None, None)}
        else:
            if is_compiled_module(module):
                module = module._orig_mod
            library = module.__module__.split('.')[0]
            pipeline_dir = module.__module__.split('.')[-2] if len(module.
                __module__.split('.')) > 2 else None
            path = module.__module__.split('.')
            is_pipeline_module = pipeline_dir in path and hasattr(pipelines,
                pipeline_dir)
            if library not in LOADABLE_CLASSES or is_pipeline_module:
                library = pipeline_dir
            class_name = module.__class__.__name__
            register_dict = {name: (library, class_name)}
        self.register_to_config(**register_dict)
        setattr(self, name, module)
