def relative_pos(self, pos, stride, pooled_pos=None, shift=1):
    """
        Build the relative positional vector between `pos` and `pooled_pos`.
        """
    if pooled_pos is None:
        pooled_pos = pos
    ref_point = pooled_pos[0] - pos[0]
    num_remove = shift * len(pooled_pos)
    max_dist = ref_point + num_remove * stride
    min_dist = pooled_pos[0] - pos[-1]
    return torch.arange(max_dist, min_dist - 1, -stride, dtype=torch.long,
        device=pos.device)
