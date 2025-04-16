def _look_adjacent(self, vectors, num_chunks_before, num_chunks_after):
    """
        Used to implement attention between consecutive chunks.

        Args:
            vectors: array of shape [batch_size, num_attention_heads, n_chunks, chunk_len, ...]
            num_chunks_before: chunks before current chunk to include in attention
            num_chunks_after: chunks after current chunk to include in attention

        Returns:
            tensor of shape [num_chunks, N * chunk_length, ...], where N = (1 + num_chunks_before + num_chunks_after).
        """
    if num_chunks_before == 0 and num_chunks_after == 0:
        return vectors
    slices = []
    for i in range(-num_chunks_before, num_chunks_after + 1):
        if i == 0:
            slices.append(vectors)
        else:
            slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :,
                :i, ...]], dim=2))
    return torch.cat(slices, dim=3)
