def count_iter_items(iterable: Iterable) ->int:
    """Consume an iterable not reading it into memory; return the number of items.

    Args:
        iterable (Iterable): Iterable object

    Returns:
        int: Number of items
    """
    counter = itertools.count()
    deque(zip(iterable, counter), maxlen=0)
    return next(counter)
