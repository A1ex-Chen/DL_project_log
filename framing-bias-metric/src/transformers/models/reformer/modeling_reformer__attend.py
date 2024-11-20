def _attend(self, query_vectors, key_vectors, value_vectors,
    sorted_bucket_idx_per_hash, attention_mask, head_mask,
    do_standard_self_attention, do_cached_attention):
    if not do_standard_self_attention:
        key_vectors = self._look_adjacent(key_vectors, self.
            num_chunks_before, self.num_chunks_after)
        value_vectors = self._look_adjacent(value_vectors, self.
            num_chunks_before, self.num_chunks_after)
    query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
    del query_vectors, key_vectors
    if not do_standard_self_attention:
        query_bucket_idx = self._split_seq_length_dim_to(
            sorted_bucket_idx_per_hash, -1, self.chunk_length, self.
            num_attention_heads)
        key_value_bucket_idx = self._look_adjacent(query_bucket_idx, self.
            num_chunks_before, self.num_chunks_after)
    elif do_cached_attention and query_key_dots.ndim > 4:
        key_value_bucket_idx = sorted_bucket_idx_per_hash
        query_bucket_idx = key_value_bucket_idx.new_ones(
            key_value_bucket_idx.shape[:-1] + (1,)) * key_value_bucket_idx.max(
            )
    elif do_cached_attention and query_key_dots.ndim <= 4:
        query_bucket_idx = (query_key_dots.shape[-1] - 1) * torch.ones_like(
            query_key_dots)[:, :, :, -1]
        key_value_bucket_idx = torch.arange(query_key_dots.shape[-1], dtype
            =torch.long, device=query_key_dots.device)[None, None, :].expand(
            query_bucket_idx.shape[:2] + (-1,))
    else:
        query_bucket_idx = key_value_bucket_idx = sorted_bucket_idx_per_hash
    if query_key_dots.dtype == torch.float16:
        self_mask_value = self.self_mask_value_float16.half()
        mask_value = self.mask_value_float16.half()
    else:
        self_mask_value = self.self_mask_value_float32
        mask_value = self.mask_value_float32
    if not do_cached_attention:
        mask = self._compute_attn_mask(query_bucket_idx,
            key_value_bucket_idx, attention_mask, query_key_dots.shape,
            do_standard_self_attention)
        if mask is not None:
            query_key_dots = torch.where(mask, query_key_dots, mask_value)
        del mask
    self_mask = torch.ne(query_bucket_idx.unsqueeze(-1),
        key_value_bucket_idx.unsqueeze(-2)).to(query_bucket_idx.device)
    query_key_dots = torch.where(self_mask, query_key_dots, self_mask_value)
    del self_mask
    logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
    attention_probs = torch.exp(query_key_dots - logits)
    del query_key_dots
    attention_probs = nn.functional.dropout(attention_probs, p=self.dropout,
        training=self.training)
    if head_mask is not None:
        attention_probs = attention_probs * head_mask
    out_vectors = torch.matmul(attention_probs, value_vectors)
    del value_vectors
    if out_vectors.ndim > 4:
        logits = logits.flatten(start_dim=2, end_dim=3).squeeze(-1)
        out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)
    return out_vectors, logits, attention_probs
