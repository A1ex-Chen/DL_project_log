@staticmethod
def _get_index(match_indices):
    """Returns batch indices, source indices, and destination indices from provided match indices."""
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in
        enumerate(match_indices)])
    src_idx = torch.cat([src for src, _ in match_indices])
    dst_idx = torch.cat([dst for _, dst in match_indices])
    return (batch_idx, src_idx), dst_idx
