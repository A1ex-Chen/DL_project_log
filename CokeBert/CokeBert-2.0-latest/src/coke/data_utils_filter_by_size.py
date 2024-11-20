def filter_by_size(indices, size_fn, max_positions, raise_exception=False):
    """
    Filter indices based on their size.

    Args:
        indices (List[int]): ordered list of dataset indices
        size_fn (callable): function that returns the size of a given index
        max_positions (tuple): filter elements larger than this size.
            Comparisons are done component-wise.
        raise_exception (bool, optional): if ``True``, raise an exception
            if any elements are filtered. Default: ``False``
    """

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
            return all(a is None or b is None or a <= b for a, b in zip(
                size_fn(idx), max_positions))
    ignored = []
    itr = collect_filtered(check_size, indices, ignored)
    for idx in itr:
        if len(ignored) > 0 and raise_exception:
            raise Exception(
                'Size of sample #{} is invalid (={}) since max_positions={}, skip this example with --skip-invalid-size-inputs-valid-test'
                .format(ignored[0], size_fn(ignored[0]), max_positions))
        yield idx
    if len(ignored) > 0:
        print(
            '| WARNING: {} samples have invalid sizes and will be skipped, max_positions={}, first few sample ids={}'
            .format(len(ignored), max_positions, ignored[:10]))
