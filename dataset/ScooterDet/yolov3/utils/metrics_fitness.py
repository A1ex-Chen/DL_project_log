def fitness(x):
    w = [0.0, 0.0, 0.1, 0.9]
    return (x[:, :4] * w).sum(1)
