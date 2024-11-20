def anchor_fitness(k):
    _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
    return (best * (best > thr).float()).mean()
