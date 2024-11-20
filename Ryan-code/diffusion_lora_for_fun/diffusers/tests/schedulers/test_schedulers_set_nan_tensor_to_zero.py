def set_nan_tensor_to_zero(t):
    t[t != t] = 0
    return t
