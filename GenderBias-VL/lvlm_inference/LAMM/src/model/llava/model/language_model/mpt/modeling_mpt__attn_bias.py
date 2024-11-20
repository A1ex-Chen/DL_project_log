@torch.no_grad()
def _attn_bias(self, device, dtype, attention_mask: Optional[torch.
    ByteTensor]=None, prefix_mask: Optional[torch.ByteTensor]=None,
    sequence_id: Optional[torch.LongTensor]=None):
    if not self._attn_bias_initialized:
        if self.attn_bias_shape:
            self.attn_bias = torch.zeros(self.attn_bias_shape, device=
                device, dtype=dtype)
            self.attn_bias = build_attn_bias(self.attn_impl, self.attn_bias,
                self.config.n_heads, self.config.max_seq_len, causal=self.
                is_causal, alibi=self.alibi, alibi_bias_max=self.alibi_bias_max
                )
        self._attn_bias_initialized = True
    if self.attn_impl == 'flash':
        return self.attn_bias, attention_mask
    if self.attn_bias is not None:
        self.attn_bias = self.attn_bias.to(dtype=dtype, device=device)
    attn_bias = self.attn_bias
    if self.prefix_lm:
        assert isinstance(attn_bias, torch.Tensor)
        assert isinstance(prefix_mask, torch.Tensor)
        attn_bias = self._apply_prefix_mask(attn_bias, prefix_mask)
    if self.attn_uses_sequence_id and sequence_id is not None:
        assert isinstance(attn_bias, torch.Tensor)
        attn_bias = self._apply_sequence_id(attn_bias, sequence_id)
    if attention_mask is not None:
        s_k = attention_mask.shape[-1]
        if attn_bias is None:
            attn_bias = torch.zeros((1, 1, 1, s_k), device=device, dtype=dtype)
        else:
            _s_k = max(0, attn_bias.size(-1) - s_k)
            attn_bias = attn_bias[:, :, :, _s_k:]
        if (prefix_mask is not None and attention_mask.shape != prefix_mask
            .shape):
            raise ValueError(
                f'attention_mask shape={attention_mask.shape} ' +
                f'and prefix_mask shape={prefix_mask.shape} are not equal.')
        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(~attention_mask.view(-1, 1, 1,
            s_k), min_val)
    return attn_bias, None
