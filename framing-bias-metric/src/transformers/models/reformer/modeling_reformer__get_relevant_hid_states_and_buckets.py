def _get_relevant_hid_states_and_buckets(self, query_vectors,
    attention_mask, num_hashes, hidden_states, past_states, past_buckets):
    hidden_states = torch.cat([past_states, hidden_states], dim=1)
    batch_size = hidden_states.shape[0]
    sequence_length = hidden_states.shape[1]
    max_bucket = self.num_buckets if isinstance(self.num_buckets, int
        ) else reduce(mul, self.num_buckets)
    increase_num_buckets = past_buckets.max() > num_hashes * max_bucket - 1
    query_buckets = self._hash_vectors(query_vectors, num_hashes,
        attention_mask, increase_num_buckets=increase_num_buckets)
    concat_buckets = torch.cat([past_buckets, query_buckets.unsqueeze(-1)],
        dim=-1)
    bucket_idx = _stable_argsort(concat_buckets, dim=-1)
    assert bucket_idx.shape == (batch_size, self.num_attention_heads,
        num_hashes, sequence_length
        ), f'bucket_idx should have shape {batch_size, self.num_attention_heads, num_hashes, sequence_length}, but has shape {bucket_idx.shape}.'
    relevant_bucket_idx = (bucket_idx == bucket_idx.shape[-1] - 1).nonzero()
    relevant_bucket_idx_chunk = self._expand_to_indices_in_relevant_chunk(
        relevant_bucket_idx, sequence_length)
    relevant_bucket_idx_chunk = bucket_idx[tuple(relevant_bucket_idx_chunk.
        transpose(0, 1))]
    bucket_idx_batch_offset = sequence_length * (batch_size * torch.arange(
        relevant_bucket_idx_chunk.shape[-1], device=hidden_states.device,
        dtype=torch.long) // relevant_bucket_idx_chunk.shape[-1])
    relevant_bucket_idx_chunk_all_batch = (relevant_bucket_idx_chunk +
        bucket_idx_batch_offset)
    hidden_states = hidden_states.reshape((-1, self.hidden_size))
    relevant_hidden_states = hidden_states.index_select(0,
        relevant_bucket_idx_chunk_all_batch)
    relevant_hidden_states = relevant_hidden_states.reshape(batch_size,
        self.num_attention_heads, -1, self.hidden_size)
    relevant_bucket_idx_chunk = relevant_bucket_idx_chunk.reshape(batch_size,
        self.num_attention_heads, num_hashes, -1)
    assert relevant_hidden_states.shape[2] == (self.num_chunks_before +
        self.num_chunks_after + 1
        ) * self.chunk_length * num_hashes, f'There should be {(self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length * num_hashes} `hidden_states`, there are {relevant_hidden_states.shape[2]} `hidden_states`.'
    assert relevant_bucket_idx_chunk.shape[-1] == (self.num_chunks_before +
        self.num_chunks_after + 1
        ) * self.chunk_length, f'There should be {(self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length} `hidden_states`, there are {relevant_bucket_idx_chunk.shape[-1]} `bucket_idx`.'
    return relevant_hidden_states, relevant_bucket_idx_chunk, query_buckets
