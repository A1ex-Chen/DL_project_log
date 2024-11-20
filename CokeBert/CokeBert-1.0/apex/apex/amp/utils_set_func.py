def set_func(mod, fn, new_fn):
    if isinstance(mod, torch.nn.backends.backend.FunctionBackend):
        mod.function_classes[fn] = new_fn
    elif isinstance(mod, dict):
        mod[fn] = new_fn
    else:
        setattr(mod, fn, new_fn)
