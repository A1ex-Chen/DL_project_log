def ids_tensor(shape, vocab_size, rng=None, name=None):
    """Creates a random int32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = global_rng
    total_dims = 1
    for dim in shape:
        total_dims *= dim
    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))
    return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()
