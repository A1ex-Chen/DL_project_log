def _sliding_chunks_matmul_attn_probs_value(self, attn_probs: torch.Tensor,
    value: torch.Tensor, window_overlap: int):
    """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
    batch_size, seq_len, num_heads, head_dim = value.size()
    assert seq_len % (window_overlap * 2) == 0
    assert attn_probs.size()[:3] == value.size()[:3]
    assert attn_probs.size(3) == 2 * window_overlap + 1
    chunks_count = seq_len // window_overlap - 1
    chunked_attn_probs = attn_probs.transpose(1, 2).reshape(batch_size *
        num_heads, seq_len // window_overlap, window_overlap, 2 *
        window_overlap + 1)
    value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len,
        head_dim)
    padded_value = F.pad(value, (0, 0, window_overlap, window_overlap),
        value=-1)
    chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 *
        window_overlap, head_dim)
    chunked_value_stride = padded_value.stride()
    chunked_value_stride = chunked_value_stride[0
        ], window_overlap * chunked_value_stride[1], chunked_value_stride[1
        ], chunked_value_stride[2]
    chunked_value = padded_value.as_strided(size=chunked_value_size, stride
        =chunked_value_stride)
    chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)
    context = torch.einsum('bcwd,bcdh->bcwh', (chunked_attn_probs,
        chunked_value))
    return context.view(batch_size, num_heads, seq_len, head_dim).transpose(
        1, 2)
