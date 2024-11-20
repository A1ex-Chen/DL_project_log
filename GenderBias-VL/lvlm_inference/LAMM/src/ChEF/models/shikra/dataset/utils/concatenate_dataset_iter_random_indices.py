def iter_random_indices():
    """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
    rng = np.random.default_rng(seed)
    while True:
        yield from (int(i) for i in rng.choice(len(lengths), size=1000, p=
            probabilities))
