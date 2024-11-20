def _len_and_dim_norm(self, vectors):
    """
        length and attention head size dim normalization
        """
    vectors = self._len_norm(vectors)
    vectors = vectors * torch.rsqrt(torch.tensor(self.attention_head_size,
        device=vectors.device, dtype=vectors.dtype))
    return vectors
