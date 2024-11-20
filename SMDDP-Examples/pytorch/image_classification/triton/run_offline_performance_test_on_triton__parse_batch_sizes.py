def _parse_batch_sizes(batch_sizes: str):
    batches = batch_sizes.split(sep=',')
    return list(map(lambda x: int(x.strip()), batches))
