def update_rotary_pos_emb_cache(self, max_seq_len, offset=0, ntk_alpha=1.0):
    seqlen = max_seq_len + offset
    if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
        base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
        self.inv_freq = 1.0 / base ** (torch.arange(0, self.dim, 2, device=
            self.inv_freq.device).float() / self.dim)
        self._seq_len_cached = max(2 * seqlen, 16)
        self._ntk_alpha_cached = ntk_alpha
        seq = torch.arange(self._seq_len_cached, device=self.inv_freq.device)
        freqs = torch.outer(seq.type_as(self.inv_freq), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        from einops import rearrange
        emb = rearrange(emb, 'n d -> 1 n 1 d')
        cos, sin = emb.cos(), emb.sin()
        self._rotary_pos_emb_cache = [cos, sin]
