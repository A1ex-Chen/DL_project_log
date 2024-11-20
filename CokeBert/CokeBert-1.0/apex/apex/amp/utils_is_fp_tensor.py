def is_fp_tensor(x):
    if is_nested(x):
        for y in x:
            if not is_fp_tensor(y):
                return False
        return True
    return compat.is_tensor_like(x) and compat.is_floating_point(x)
