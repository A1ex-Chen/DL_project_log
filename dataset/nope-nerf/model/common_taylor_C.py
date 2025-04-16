def taylor_C(x, nth=10):
    ans = torch.zeros_like(x)
    denom = 1.0
    for i in range(nth + 1):
        denom *= (2 * i + 2) * (2 * i + 3)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans
