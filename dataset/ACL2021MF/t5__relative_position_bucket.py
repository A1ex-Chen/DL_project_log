@staticmethod
def _relative_position_bucket(relative_position, bidirectional=True,
    num_buckets=32, max_distance=128):
    """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
    ret = 0
    n = -relative_position
    if bidirectional:
        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)
    else:
        n = torch.max(n, torch.zeros_like(n))
    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log
        (max_distance / max_exact) * (num_buckets - max_exact)).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, 
        num_buckets - 1))
    ret += torch.where(is_small, n, val_if_large)
    return ret
