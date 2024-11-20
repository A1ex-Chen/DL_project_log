def forward(self, x, past_key_value=None, attn_bias=None, attention_mask=
    None, is_causal=True, needs_weights=False):
    qkv = self.Wqkv(x)
    if self.clip_qkv:
        qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
    query, key, value = qkv.split([self.d_model, self.head_dim, self.
        head_dim], dim=2)
    key_padding_mask = attention_mask
    if self.qk_ln:
        dtype = query.dtype
        query = self.q_ln(query).to(dtype)
        key = self.k_ln(key).to(dtype)
    context, attn_weights, past_key_value = self.attn_fn(query, key, value,
        self.n_heads, past_key_value=past_key_value, softmax_scale=self.
        softmax_scale, attn_bias=attn_bias, key_padding_mask=
        key_padding_mask, is_causal=is_causal, dropout_p=self.
        attn_dropout_p, training=self.training, needs_weights=needs_weights,
        multiquery=True)
    return self.out_proj(context), attn_weights, past_key_value
