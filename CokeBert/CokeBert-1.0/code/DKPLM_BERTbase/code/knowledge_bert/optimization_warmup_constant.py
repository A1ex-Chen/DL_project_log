def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0