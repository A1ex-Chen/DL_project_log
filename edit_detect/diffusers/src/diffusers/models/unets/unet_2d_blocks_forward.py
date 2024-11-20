def forward(self, hidden_states: torch.Tensor, encoder_hidden_states:
    Optional[torch.Tensor]=None, emb: Optional[torch.Tensor]=None,
    attention_mask: Optional[torch.Tensor]=None, cross_attention_kwargs:
    Optional[Dict[str, Any]]=None, encoder_attention_mask: Optional[torch.
    Tensor]=None) ->torch.Tensor:
    cross_attention_kwargs = (cross_attention_kwargs if 
        cross_attention_kwargs is not None else {})
    if cross_attention_kwargs.get('scale', None) is not None:
        logger.warning(
            'Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.'
            )
    if self.add_self_attention:
        norm_hidden_states = self.norm1(hidden_states, emb)
        height, weight = norm_hidden_states.shape[2:]
        norm_hidden_states = self._to_3d(norm_hidden_states, height, weight)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=
            None, attention_mask=attention_mask, **cross_attention_kwargs)
        attn_output = self._to_4d(attn_output, height, weight)
        hidden_states = attn_output + hidden_states
    norm_hidden_states = self.norm2(hidden_states, emb)
    height, weight = norm_hidden_states.shape[2:]
    norm_hidden_states = self._to_3d(norm_hidden_states, height, weight)
    attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=
        encoder_hidden_states, attention_mask=attention_mask if 
        encoder_hidden_states is None else encoder_attention_mask, **
        cross_attention_kwargs)
    attn_output = self._to_4d(attn_output, height, weight)
    hidden_states = attn_output + hidden_states
    return hidden_states
