def _gather_by_expansion(self, vectors, idxs, num_hashes):
    """
        expand dims of idxs and vectors for all hashes and gather
        """
    expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.
        attention_head_size)
    vectors = vectors.repeat(1, 1, num_hashes, 1)
    return torch.gather(vectors, 2, expanded_idxs)
