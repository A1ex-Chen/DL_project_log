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
