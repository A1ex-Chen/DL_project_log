def set_nan_tensor_to_zero(t):
    return t.at[t != t].set(0)
