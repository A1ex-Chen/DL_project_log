def flat_dist_call(tensors, call, extra_args=None):
    buckets = split_by_type(tensors)
    for tp in buckets:
        bucket = buckets[tp]
        apply_flat_dist_call(bucket, call, extra_args)
