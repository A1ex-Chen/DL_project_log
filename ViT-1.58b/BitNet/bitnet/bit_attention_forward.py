def forward(self, query: Tensor, key: Tensor, value: Tensor, need_weights:
    bool=False, is_causal: bool=False, average_attn_weights: bool=False
    ) ->Tuple[Tensor, Optional[Tensor]]:
    q: Tensor = self.q_proj(query)
    k: Tensor = self.k_proj(key)
    v: Tensor = self.v_proj(value)
    q = rearrange(q, 'b n (h d) -> b n h d', h=self.query_heads)
    k = rearrange(k, 'b n (h d) -> b n h d', h=self.kv_heads)
    v = rearrange(v, 'b n (h d) -> b n h d', h=self.kv_heads)
    x, attn_weights = scaled_dot_product_gqa(query=q, key=k, value=v,
        is_causal=is_causal, need_weights=need_weights,
        average_attn_weights=average_attn_weights, force_grouped=False)
    x = rearrange(x, 'b n h d -> b n (h d)')
    if self.layer_norm:
        assert self.norm is not None
        x = self.norm(x)
    x = self.out_proj(x)
    return x, attn_weights
