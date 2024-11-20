def get_bucket_sizes(model: torch.nn.Module, cap_size: int) ->List[int]:
    """
    Inputs: Pytorch model and a bucket size cap
    Outputs: list of bucket sizes in Megabytes.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    bucket_cap_mb = cap_size
    bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)
    bucket_size_limits = [dist._DEFAULT_FIRST_BUCKET_BYTES, bucket_bytes_cap]
    bucket_indices, per_bucket_size_limits = (dist.
        _compute_bucket_assignment_by_size(params, bucket_size_limits, [
        False] * len(params)))
    bucket_sizes = []
    bucket_indices_backward = bucket_indices[::-1]
    params_in_buckets = tree_map(lambda idx: list(model.parameters())[idx],
        bucket_indices_backward)
    for bucket in params_in_buckets:
        size_bytes = sum(p.numel() * p.element_size() for p in bucket)
        size_mb = round(size_bytes / 1024 / 1024, 3)
        bucket_sizes.append(size_mb)
    return bucket_sizes
