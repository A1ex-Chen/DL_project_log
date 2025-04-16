def add_wrapper(mod, fn_name):
    assert isfunc(mod, fn_name)
    func = getattr(mod, fn_name)
    s = hasattr(mod, 'extra_repr') and type(mod
        ) is not torch.jit.ScriptModule and type(mod
        ) is not torch.jit.TopLevelTracedModule

    def wrapper_func(*args, **kwargs):
        stack = traceback.extract_stack()
        nvtx.range_push(traceMarker(stack))
        if s:
            m = modMarker(mod, fn_name, args)
            nvtx.range_push(m)
        cadena = argMarker(mod, fn_name, args, kwargs)
        nvtx.range_push(cadena)
        result = func(*args, **kwargs)
        nvtx.range_pop()
        if s:
            nvtx.range_pop()
        nvtx.range_pop()
        return result
    setattr(mod, fn_name, wrapper_func)
