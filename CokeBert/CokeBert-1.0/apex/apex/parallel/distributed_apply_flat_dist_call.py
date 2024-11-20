def apply_flat_dist_call(bucket, call, extra_args=None):
    coalesced = flatten(bucket)
    if extra_args is not None:
        call(coalesced, *extra_args)
    else:
        call(coalesced)
    if call is dist.all_reduce:
        coalesced /= dist.get_world_size()
    for buf, synced in zip(bucket, unflatten(coalesced, bucket)):
        buf.copy_(synced)
