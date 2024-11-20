def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t).contiguous()
    return out.reshape(b, *((1,) * (len(x_shape) - 1))).contiguous()
