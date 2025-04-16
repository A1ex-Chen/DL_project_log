def _set_cos_sin_cache(self, seq_len, device, dtype):
    self.max_seq_len_cached = seq_len
    if seq_len > self.max_position_embeddings:
        base = self.base * (self.scaling_factor * seq_len / self.
            max_position_embeddings - (self.scaling_factor - 1)) ** (self.
            dim / (self.dim - 2))
        inv_freq = 1.0 / base ** (torch.arange(0, self.dim, 2).float().to(
            device) / self.dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
    t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.
        inv_freq.dtype)
    freqs = torch.einsum('i,j->ij', t, self.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer('cos_cached', emb.cos()[None, None, :, :].to(dtype
        ), persistent=False)
    self.register_buffer('sin_cached', emb.sin()[None, None, :, :].to(dtype
        ), persistent=False)
