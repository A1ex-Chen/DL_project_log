def __call__(self, hidden_states, deterministic=True):
    hidden_states = self.proj(hidden_states)
    hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=2)
    return self.dropout_layer(hidden_linear * nn.gelu(hidden_gelu),
        deterministic=deterministic)
