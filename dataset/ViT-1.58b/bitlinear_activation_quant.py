def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-05)
    y = (x * scale).round().clamp_(-128, 127)
    return y, scale
