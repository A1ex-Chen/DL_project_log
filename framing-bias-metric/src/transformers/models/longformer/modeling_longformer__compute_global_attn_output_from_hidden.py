def _compute_global_attn_output_from_hidden(self, hidden_states,
    max_num_global_attn_indices, is_local_index_global_attn_nonzero,
    is_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero,
    is_index_masked):
    seq_len, batch_size = hidden_states.shape[:2]
    global_attn_hidden_states = hidden_states.new_zeros(
        max_num_global_attn_indices, batch_size, self.embed_dim)
    global_attn_hidden_states[is_local_index_global_attn_nonzero[::-1]
        ] = hidden_states[is_index_global_attn_nonzero[::-1]]
    global_query_vectors_only_global = self.query_global(
        global_attn_hidden_states)
    global_key_vectors = self.key_global(hidden_states)
    global_value_vectors = self.value_global(hidden_states)
    global_query_vectors_only_global /= math.sqrt(self.head_dim)
    global_query_vectors_only_global = (global_query_vectors_only_global.
        contiguous().view(max_num_global_attn_indices, batch_size * self.
        num_heads, self.head_dim).transpose(0, 1))
    global_key_vectors = global_key_vectors.contiguous().view(-1, 
        batch_size * self.num_heads, self.head_dim).transpose(0, 1)
    global_value_vectors = global_value_vectors.contiguous().view(-1, 
        batch_size * self.num_heads, self.head_dim).transpose(0, 1)
    global_attn_scores = torch.bmm(global_query_vectors_only_global,
        global_key_vectors.transpose(1, 2))
    assert list(global_attn_scores.size()) == [batch_size * self.num_heads,
        max_num_global_attn_indices, seq_len
        ], f'global_attn_scores have the wrong size. Size should be {batch_size * self.num_heads, max_num_global_attn_indices, seq_len}, but is {global_attn_scores.size()}.'
    global_attn_scores = global_attn_scores.view(batch_size, self.num_heads,
        max_num_global_attn_indices, seq_len)
    global_attn_scores[is_local_index_no_global_attn_nonzero[0], :,
        is_local_index_no_global_attn_nonzero[1], :] = -10000.0
    global_attn_scores = global_attn_scores.masked_fill(is_index_masked[:,
        None, None, :], -10000.0)
    global_attn_scores = global_attn_scores.view(batch_size * self.
        num_heads, max_num_global_attn_indices, seq_len)
    global_attn_probs_float = F.softmax(global_attn_scores, dim=-1, dtype=
        torch.float32)
    global_attn_probs = F.dropout(global_attn_probs_float.type_as(
        global_attn_scores), p=self.dropout, training=self.training)
    global_attn_output = torch.bmm(global_attn_probs, global_value_vectors)
    assert list(global_attn_output.size()) == [batch_size * self.num_heads,
        max_num_global_attn_indices, self.head_dim
        ], f'global_attn_output tensor has the wrong size. Size should be {batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim}, but is {global_attn_output.size()}.'
    global_attn_probs = global_attn_probs.view(batch_size, self.num_heads,
        max_num_global_attn_indices, seq_len)
    global_attn_output = global_attn_output.view(batch_size, self.num_heads,
        max_num_global_attn_indices, self.head_dim)
    return global_attn_output, global_attn_probs
