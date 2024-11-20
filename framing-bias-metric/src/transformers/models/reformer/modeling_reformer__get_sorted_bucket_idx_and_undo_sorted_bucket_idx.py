def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(self, sequence_length,
    buckets, num_hashes):
    with torch.no_grad():
        sorted_bucket_idx = _stable_argsort(buckets, dim=-1)
        indices = torch.arange(sorted_bucket_idx.shape[-1], device=buckets.
            device).view(1, 1, -1).expand(sorted_bucket_idx.shape)
        undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.
            size())
        undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)
    return sorted_bucket_idx, undo_sorted_bucket_idx
