def _get_output_for_continuous_inputs(self, hidden_states, residual,
    batch_size, height, width, inner_dim):
    if not self.use_linear_projection:
        hidden_states = hidden_states.reshape(batch_size, height, width,
            inner_dim).permute(0, 3, 1, 2).contiguous()
        hidden_states = self.proj_out(hidden_states)
    else:
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, height, width,
            inner_dim).permute(0, 3, 1, 2).contiguous()
    output = hidden_states + residual
    return output
