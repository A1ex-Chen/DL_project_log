def _shard_iterator_dataloader_worker(iterable):
    worker_info = data.get_worker_info()
    if worker_info is None or worker_info.num_workers == 1:
        yield from iterable
    else:
        yield from itertools.islice(iterable, worker_info.id, None,
            worker_info.num_workers)
