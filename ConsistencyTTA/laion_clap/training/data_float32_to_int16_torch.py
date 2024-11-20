def float32_to_int16_torch(x):
    x = torch.clamp(x, min=-1.0, max=1.0)
    return (x * 32767.0).type(torch.int16)
