def int16_to_float32_torch(x):
    return (x / 32767.0).type(torch.float32)
