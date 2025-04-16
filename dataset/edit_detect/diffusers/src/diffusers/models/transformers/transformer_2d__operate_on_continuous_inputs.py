def _operate_on_continuous_inputs(self, hidden_states):
    batch, _, height, width = hidden_states.shape
    hidden_states = self.norm(hidden_states)
    if not self.use_linear_projection:
        hidden_states = self.proj_in(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, 
            height * width, inner_dim)
    else:
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, 
            height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)
    return hidden_states, inner_dim
