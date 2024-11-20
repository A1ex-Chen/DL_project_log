def _interleave_dataset_index(*, lengths: List[int], probabilities:
    Optional[List[float]]=None, seed: Optional[int]=None, stopping_strategy:
    Literal['first_exhausted', 'all_exhausted']='first_exhausted'):
    if probabilities is not None and 0 in probabilities:
        assert stopping_strategy == 'first_exhausted', 'you will meet a Infinite loop'
    offsets = np.cumsum([0] + lengths[:-1])
    oversampling = stopping_strategy == 'all_exhausted'
    if probabilities is None and not oversampling:
        indices = (offsets.reshape(1, -1) + np.arange(min(lengths)).reshape
            (-1, 1)).flatten().tolist()
    elif probabilities is None:
        indices = np.mod(np.arange(max(lengths)).reshape(-1, 1), np.array(
            lengths).reshape(1, -1))
        indices = (indices + offsets).flatten().tolist()
    else:
        is_exhausted = np.full(len(lengths), False)
        bool_strategy_func = np.all if oversampling else np.any

        def iter_random_indices():
            """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
            rng = np.random.default_rng(seed)
            while True:
                yield from (int(i) for i in rng.choice(len(lengths), size=
                    1000, p=probabilities))
        current_index = [0] * len(lengths)
        indices = []
        for source_idx in iter_random_indices():
            if bool_strategy_func(is_exhausted):
                break
            indices.append(current_index[source_idx] + offsets[source_idx])
            current_index[source_idx] += 1
            if current_index[source_idx] >= lengths[source_idx]:
                is_exhausted[source_idx] = True
                current_index[source_idx] = 0
    return indices
