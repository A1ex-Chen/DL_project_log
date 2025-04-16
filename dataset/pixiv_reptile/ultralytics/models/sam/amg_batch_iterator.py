def batch_iterator(batch_size: int, *args) ->Generator[List[Any], None, None]:
    """Yield batches of data from the input arguments."""
    assert args and all(len(a) == len(args[0]) for a in args
        ), 'Batched iteration must have same-size inputs.'
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0
        )
    for b in range(n_batches):
        yield [arg[b * batch_size:(b + 1) * batch_size] for arg in args]
