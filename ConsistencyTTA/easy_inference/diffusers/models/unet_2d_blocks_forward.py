def forward(self, hidden_states: torch.FloatTensor, encoder_hidden_states:
    Optional[torch.FloatTensor]=None, emb: Optional[torch.FloatTensor]=None,
    attention_mask: Optional[torch.FloatTensor]=None,
    cross_attention_kwargs: Optional[Dict[str, Any]]=None,
    encoder_attention_mask: Optional[torch.FloatTensor]=None):
    cross_attention_kwargs = (cross_attention_kwargs if 
        cross_attention_kwargs is not None else {})
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
