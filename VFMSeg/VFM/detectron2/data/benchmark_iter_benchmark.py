def iter_benchmark(iterator, num_iter: int, warmup: int=5, max_time_seconds:
    float=60) ->Tuple[float, List[float]]:
    """
    Benchmark an iterator/iterable for `num_iter` iterations with an extra
    `warmup` iterations of warmup.
    End early if `max_time_seconds` time is spent on iterations.

    Returns:
        float: average time (seconds) per iteration
        list[float]: time spent on each iteration. Sometimes useful for further analysis.
    """
    num_iter, warmup = int(num_iter), int(warmup)
    iterator = iter(iterator)
    for _ in range(warmup):
        next(iterator)
    timer = Timer()
    all_times = []
    for curr_iter in tqdm.trange(num_iter):
        start = timer.seconds()
        if start > max_time_seconds:
            num_iter = curr_iter
            break
        next(iterator)
        all_times.append(timer.seconds() - start)
    avg = timer.seconds() / num_iter
    return avg, all_times
