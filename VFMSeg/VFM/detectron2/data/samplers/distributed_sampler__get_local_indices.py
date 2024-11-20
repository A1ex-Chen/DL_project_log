@staticmethod
def _get_local_indices(total_size, world_size, rank):
    shard_size = total_size // world_size
    left = total_size % world_size
    shard_sizes = [(shard_size + int(r < left)) for r in range(world_size)]
    begin = sum(shard_sizes[:rank])
    end = min(sum(shard_sizes[:rank + 1]), total_size)
    return range(begin, end)
