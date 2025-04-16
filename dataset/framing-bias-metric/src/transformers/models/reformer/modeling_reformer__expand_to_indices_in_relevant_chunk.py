def _expand_to_indices_in_relevant_chunk(self, indices, sequence_length):
    start_indices_chunk = (indices[:, -1] // self.chunk_length - self.
        num_chunks_before) * self.chunk_length
    total_chunk_size = self.chunk_length * (1 + self.num_chunks_before +
        self.num_chunks_after)
    expanded_start_indices = start_indices_chunk.unsqueeze(-1).expand(indices
        .shape[0], total_chunk_size)
    chunk_sequence_indices = expanded_start_indices + torch.arange(
        total_chunk_size, device=indices.device, dtype=torch.long).unsqueeze(0
        ).expand(indices.shape[0], total_chunk_size)
    chunk_sequence_indices = chunk_sequence_indices.flatten() % sequence_length
    indices = indices.unsqueeze(1).expand((indices.shape[0],
        total_chunk_size, -1)).flatten(0, 1).clone()
    indices[:, -1] = chunk_sequence_indices
    return indices
