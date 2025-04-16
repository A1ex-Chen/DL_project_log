def register_model(fn):
    module_name_split = fn.__module__.split('.')
    model_name = module_name_split[-1]
    _model_entrypoints[model_name] = fn
    return fn
