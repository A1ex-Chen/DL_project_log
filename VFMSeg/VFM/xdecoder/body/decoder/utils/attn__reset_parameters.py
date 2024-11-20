def _reset_parameters(self):
    if self._qkv_same_embed_dim:
        xavier_uniform_(self.in_proj_weight)
    else:
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)
    if self.in_proj_bias is not None:
        constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)
    if self.bias_k is not None:
        xavier_normal_(self.bias_k)
    if self.bias_v is not None:
        xavier_normal_(self.bias_v)
