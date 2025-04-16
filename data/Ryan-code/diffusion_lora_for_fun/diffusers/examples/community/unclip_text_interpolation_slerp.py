def slerp(val, low, high):
    """
    Find the interpolation point between the 'low' and 'high' values for the given 'val'. See https://en.wikipedia.org/wiki/Slerp for more details on the topic.
    """
    low_norm = low / torch.norm(low)
    high_norm = high / torch.norm(high)
    omega = torch.acos(low_norm * high_norm)
    so = torch.sin(omega)
    res = torch.sin((1.0 - val) * omega) / so * low + torch.sin(val * omega
        ) / so * high
    return res
