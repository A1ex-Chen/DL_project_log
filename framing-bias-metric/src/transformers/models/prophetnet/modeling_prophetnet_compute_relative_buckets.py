def compute_relative_buckets(num_buckets, max_distance, relative_positions,
    is_bidirectional=False):
    """
    This function computes individual parts of the relative position buckets. For more detail, see paper.
    """
    inv_relative_positions = -relative_positions
    rel_positions_bucket = 0
    if is_bidirectional:
        num_buckets = num_buckets // 2
        rel_positions_bucket = rel_positions_bucket + torch.lt(
            inv_relative_positions, torch.zeros_like(inv_relative_positions)
            ).int() * num_buckets
        inv_relative_positions = torch.abs(inv_relative_positions)
    else:
        inv_relative_positions = torch.max(inv_relative_positions, torch.
            zeros_like(inv_relative_positions))
    max_exact = num_buckets // 2
    is_small = torch.lt(inv_relative_positions, max_exact)
    val_if_large = max_exact + torch.log(inv_relative_positions.float() /
        max_exact) / math.log(max_distance / max_exact) * (num_buckets -
        max_exact)
    val_if_large = torch.min(val_if_large, torch.ones_like(val_if_large) *
        (num_buckets - 1)).int()
    rel_positions_bucket = rel_positions_bucket + torch.where(is_small,
        inv_relative_positions.int(), val_if_large)
    return rel_positions_bucket
