def _compute_attn_mask(self, query_indices, key_indices, attention_mask,
    query_key_dots_shape, do_standard_self_attention):
    if attention_mask is not None:
        attention_mask = attention_mask.to(torch.uint8)[:, None, :]
        if not do_standard_self_attention:
            attention_mask = self._split_seq_length_dim_to(attention_mask, 
                -1, self.chunk_length, 1)
            attention_mask = self._look_adjacent(attention_mask, self.
                num_chunks_before, self.num_chunks_after)
        attention_mask = attention_mask.unsqueeze(-2).expand(
            query_key_dots_shape)
    if self.is_decoder is True:
        causal_mask = torch.ge(query_indices.unsqueeze(-1), key_indices.
            unsqueeze(-2)).to(query_indices.device)
        if attention_mask is not None:
            attention_mask = causal_mask * attention_mask
        else:
            attention_mask = causal_mask
    return attention_mask
