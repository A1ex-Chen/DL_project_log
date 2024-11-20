def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
    res_hidden_states = res_hidden_states_tuple[-1]
    hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
    for resnet in self.resnets:
        hidden_states = resnet(hidden_states)
    return hidden_states
