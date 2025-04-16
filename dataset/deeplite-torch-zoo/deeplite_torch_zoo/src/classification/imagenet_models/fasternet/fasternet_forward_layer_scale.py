def forward_layer_scale(self, x: Tensor) ->Tensor:
    shortcut = x
    x = self.spatial_mixing(x)
    x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(
        -1) * self.mlp(x))
    return x
