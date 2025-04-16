def forward(self, q: Tensor, k: Tensor, v: Tensor) ->Tensor:
    """Compute the attention output given the input query, key, and value tensors."""
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)
    q = self._separate_heads(q, self.num_heads)
    k = self._separate_heads(k, self.num_heads)
    v = self._separate_heads(v, self.num_heads)
    _, _, _, c_per_head = q.shape
    attn = q @ k.permute(0, 1, 3, 2)
    attn = attn / math.sqrt(c_per_head)
    attn = torch.softmax(attn, dim=-1)
    out = attn @ v
    out = self._recombine_heads(out)
    return self.out_proj(out)
