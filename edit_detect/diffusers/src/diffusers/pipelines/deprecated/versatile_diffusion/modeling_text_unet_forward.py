def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor]
    =None, encoder_hidden_states: Optional[torch.Tensor]=None,
    attention_mask: Optional[torch.Tensor]=None, cross_attention_kwargs:
    Optional[Dict[str, Any]]=None, encoder_attention_mask: Optional[torch.
    Tensor]=None) ->torch.Tensor:
    cross_attention_kwargs = (cross_attention_kwargs if 
        cross_attention_kwargs is not None else {})
    if cross_attention_kwargs.get('scale', None) is not None:
        logger.warning(
            'Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.'
            )
    if attention_mask is None:
        mask = (None if encoder_hidden_states is None else
            encoder_attention_mask)
    else:
        mask = attention_mask
    hidden_states = self.resnets[0](hidden_states, temb)
    for attn, resnet in zip(self.attentions, self.resnets[1:]):
        hidden_states = attn(hidden_states, encoder_hidden_states=
            encoder_hidden_states, attention_mask=mask, **
            cross_attention_kwargs)
        hidden_states = resnet(hidden_states, temb)
    return hidden_states
