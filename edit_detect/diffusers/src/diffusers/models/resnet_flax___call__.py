def __call__(self, hidden_states, temb, deterministic=True):
    residual = hidden_states
    hidden_states = self.norm1(hidden_states)
    hidden_states = nn.swish(hidden_states)
    hidden_states = self.conv1(hidden_states)
    temb = self.time_emb_proj(nn.swish(temb))
    temb = jnp.expand_dims(jnp.expand_dims(temb, 1), 1)
    hidden_states = hidden_states + temb
    hidden_states = self.norm2(hidden_states)
    hidden_states = nn.swish(hidden_states)
    hidden_states = self.dropout(hidden_states, deterministic)
    hidden_states = self.conv2(hidden_states)
    if self.conv_shortcut is not None:
        residual = self.conv_shortcut(residual)
    return hidden_states + residual
