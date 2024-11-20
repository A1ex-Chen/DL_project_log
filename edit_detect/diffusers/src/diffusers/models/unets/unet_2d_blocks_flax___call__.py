def __call__(self, hidden_states, temb, encoder_hidden_states,
    deterministic=True):
    hidden_states = self.resnets[0](hidden_states, temb)
    for attn, resnet in zip(self.attentions, self.resnets[1:]):
        hidden_states = attn(hidden_states, encoder_hidden_states,
            deterministic=deterministic)
        hidden_states = resnet(hidden_states, temb, deterministic=deterministic
            )
    return hidden_states
