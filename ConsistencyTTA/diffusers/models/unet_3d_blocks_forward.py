def forward(self, hidden_states, res_hidden_states_tuple, temb=None,
    upsample_size=None, num_frames=1):
    for resnet, temp_conv in zip(self.resnets, self.temp_convs):
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        hidden_states = resnet(hidden_states, temb)
        hidden_states = temp_conv(hidden_states, num_frames=num_frames)
    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size)
    return hidden_states
