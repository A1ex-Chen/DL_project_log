def _reset_parameters(self):
    """Reset module parameters."""
    constant_(self.sampling_offsets.weight.data, 0.0)
    thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.
        pi / self.n_heads)
    grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
    grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
        self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
    for i in range(self.n_points):
        grid_init[:, :, i, :] *= i + 1
    with torch.no_grad():
        self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
    constant_(self.attention_weights.weight.data, 0.0)
    constant_(self.attention_weights.bias.data, 0.0)
    xavier_uniform_(self.value_proj.weight.data)
    constant_(self.value_proj.bias.data, 0.0)
    xavier_uniform_(self.output_proj.weight.data)
    constant_(self.output_proj.bias.data, 0.0)
