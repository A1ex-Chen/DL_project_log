def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states,
    timestep, added_cond_kwargs):
    batch_size = hidden_states.shape[0]
    hidden_states = self.pos_embed(hidden_states)
    embedded_timestep = None
    if self.adaln_single is not None:
        if self.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError(
                '`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`.'
                )
        timestep, embedded_timestep = self.adaln_single(timestep,
            added_cond_kwargs, batch_size=batch_size, hidden_dtype=
            hidden_states.dtype)
    if self.caption_projection is not None:
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1,
            hidden_states.shape[-1])
    return hidden_states, encoder_hidden_states, timestep, embedded_timestep
