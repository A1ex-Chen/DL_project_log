def check_size(idx):
    if isinstance(max_positions, float) or isinstance(max_positions, int):
        return size_fn(idx) <= max_positions
    elif isinstance(max_positions, dict):
        idx_size = size_fn(idx)
        assert isinstance(idx_size, dict)
        intersect_keys = set(max_positions.keys()) & set(idx_size.keys())
        return all(idx_size[key] <= max_positions[key] for key in
            intersect_keys)
    else:
        return all(a is None or b is None or a <= b for a, b in zip(size_fn
            (idx), max_positions))
