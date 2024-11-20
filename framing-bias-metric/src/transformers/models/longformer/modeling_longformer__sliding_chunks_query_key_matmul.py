def _sliding_chunks_query_key_matmul(self, query: torch.Tensor, key: torch.
    Tensor, window_overlap: int):
    """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This
        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer) with an
        overlap of size window_overlap
        """
    batch_size, seq_len, num_heads, head_dim = query.size()
    assert seq_len % (window_overlap * 2
        ) == 0, f'Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}'
    assert query.size() == key.size()
    chunks_count = seq_len // window_overlap - 1
    query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len,
        head_dim)
    key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim
        )
    chunked_query = self._chunk(query, window_overlap)
    chunked_key = self._chunk(key, window_overlap)
    chunked_attention_scores = torch.einsum('bcxd,bcyd->bcxy', (
        chunked_query, chunked_key))
    diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
        chunked_attention_scores, padding=(0, 0, 0, 1))
    diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty((
        batch_size * num_heads, chunks_count + 1, window_overlap, 
        window_overlap * 2 + 1))
    diagonal_attention_scores[:, :-1, :, window_overlap:
        ] = diagonal_chunked_attention_scores[:, :, :window_overlap, :
        window_overlap + 1]
    diagonal_attention_scores[:, -1, :, window_overlap:
        ] = diagonal_chunked_attention_scores[:, -1, window_overlap:, :
        window_overlap + 1]
    diagonal_attention_scores[:, 1:, :, :window_overlap
        ] = diagonal_chunked_attention_scores[:, :, -(window_overlap + 1):-
        1, window_overlap + 1:]
    diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap
        ] = diagonal_chunked_attention_scores[:, 0, :window_overlap - 1, 1 -
        window_overlap:]
    diagonal_attention_scores = diagonal_attention_scores.view(batch_size,
        num_heads, seq_len, 2 * window_overlap + 1).transpose(2, 1)
    self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
    return diagonal_attention_scores
