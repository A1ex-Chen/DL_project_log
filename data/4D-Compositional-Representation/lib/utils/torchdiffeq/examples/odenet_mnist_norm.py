def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)
