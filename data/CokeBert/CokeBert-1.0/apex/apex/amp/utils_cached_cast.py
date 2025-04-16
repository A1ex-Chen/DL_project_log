def cached_cast(cast_fn, x, cache):
    if is_nested(x):
        return type(x)([cached_cast(y) for y in x])
    if x in cache:
        cached_x = cache[x]
        if x.requires_grad and cached_x.requires_grad:
            if cached_x.grad_fn.next_functions[1][0].variable is not x:
                raise RuntimeError(
                    "x and cache[x] both require grad, but x is not cache[x]'s parent.  This is likely an error."
                    )
        if torch.is_grad_enabled(
            ) and x.requires_grad != cached_x.requires_grad:
            del cache[x]
        else:
            return cached_x
    casted_x = cast_fn(x)
    cache[x] = casted_x
    return casted_x
