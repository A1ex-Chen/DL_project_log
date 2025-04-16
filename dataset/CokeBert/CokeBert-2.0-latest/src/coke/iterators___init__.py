def __init__(self, iterable, num_shards, shard_id, fill_value=None):
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError('shard_id must be between 0 and num_shards')
    self._sharded_len = len(iterable) // num_shards
    if len(iterable) % num_shards > 0:
        self._sharded_len += 1
    self.itr = itertools.zip_longest(range(self._sharded_len), itertools.
        islice(iterable, shard_id, len(iterable), num_shards), fillvalue=
        fill_value)
