def forward(self, hidden_states, temb=None, encoder_hidden_states=None,
    attention_mask=None, cross_attention_kwargs=None):
    cross_attention_kwargs = (cross_attention_kwargs if 
        cross_attention_kwargs is not None else {})
    hidden_states = self.resnets[0](hidden_states, temb)
    for attn, resnet in zip(self.attentions, self.resnets[1:]):
        hidden_states = attn(hidden_states, encoder_hidden_states=
            encoder_hidden_states, attention_mask=attention_mask, **
            cross_attention_kwargs)
        hidden_states = resnet(hidden_states, temb)
    return hidden_states
