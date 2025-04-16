def transpose_for_scores(self, projection: torch.Tensor) ->torch.Tensor:
    new_projection_shape = projection.size()[:-1] + (self.num_heads, -1)
    new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
    return new_projection
