def forward(self, x: torch.Tensor, past_key_value: Optional[Tuple[torch.
    Tensor]]=None, attn_bias: Optional[torch.Tensor]=None, attention_mask:
    Optional[torch.ByteTensor]=None, is_causal: bool=True) ->Tuple[torch.
    Tensor, Optional[Tuple[torch.Tensor]]]:
    a = self.norm_1(x)
    b, attn_weights, past_key_value = self.attn(a, past_key_value=
        past_key_value, attn_bias=attn_bias, attention_mask=attention_mask,
        is_causal=is_causal)
    x = x + self.resid_attn_dropout(b)
    m = self.norm_2(x)
    n = self.ffn(m)
    x = x + self.resid_ffn_dropout(n)
    return x, attn_weights, past_key_value
