def dequantize(x, dequantize=True, reverse=False, alpha=0.1):
    with torch.no_grad():
        if reverse:
            return torch.ceil_(torch.sigmoid(x) * 255)
        if dequantize:
            x += torch.zeros_like(x).uniform_(0, 1)
        p = alpha / 2 + (1 - alpha) * x / 256
        return torch.log(p) - torch.log(1 - p)
