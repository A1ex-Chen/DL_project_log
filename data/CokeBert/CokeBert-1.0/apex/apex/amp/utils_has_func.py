def has_func(mod, fn):
    if isinstance(mod, torch.nn.backends.backend.FunctionBackend):
        return fn in mod.function_classes
    elif isinstance(mod, dict):
        return fn in mod
    else:
        return hasattr(mod, fn)
