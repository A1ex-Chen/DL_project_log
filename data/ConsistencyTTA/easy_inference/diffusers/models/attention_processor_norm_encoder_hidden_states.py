def norm_encoder_hidden_states(self, encoder_hidden_states):
    assert self.norm_cross is not None, 'self.norm_cross must be defined to call self.norm_encoder_hidden_states'
    if isinstance(self.norm_cross, nn.LayerNorm):
        encoder_hidden_states = self.norm_cross(encoder_hidden_states)
    elif isinstance(self.norm_cross, nn.GroupNorm):
        encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
    else:
        assert False
    return encoder_hidden_states
