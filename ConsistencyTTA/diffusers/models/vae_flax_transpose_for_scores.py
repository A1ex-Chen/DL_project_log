def transpose_for_scores(self, projection):
    new_projection_shape = projection.shape[:-1] + (self.num_heads, -1)
    new_projection = projection.reshape(new_projection_shape)
    new_projection = jnp.transpose(new_projection, (0, 2, 1, 3))
    return new_projection
