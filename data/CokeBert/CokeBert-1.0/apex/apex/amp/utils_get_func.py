def get_func(mod, fn):
    if isinstance(mod, torch.nn.backends.backend.FunctionBackend):
        return mod.function_classes[fn]
    elif isinstance(mod, dict):
        return mod[fn]
    else:
        return getattr(mod, fn)
