def set_nan_tensor_to_zero(t):
    device = t.device
    if device.type == 'mps':
        t = t.to('cpu')
    t[t != t] = 0
    return t.to(device)
