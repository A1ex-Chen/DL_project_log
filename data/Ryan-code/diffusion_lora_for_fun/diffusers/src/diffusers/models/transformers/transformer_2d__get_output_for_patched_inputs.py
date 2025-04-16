def _get_output_for_patched_inputs(self, hidden_states, timestep,
    class_labels, embedded_timestep, height=None, width=None):
    if self.config.norm_type != 'ada_norm_single':
        conditioning = self.transformer_blocks[0].norm1.emb(timestep,
            class_labels, hidden_dtype=hidden_states.dtype)
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]
            ) + shift[:, None]
        hidden_states = self.proj_out_2(hidden_states)
    elif self.config.norm_type == 'ada_norm_single':
        shift, scale = (self.scale_shift_table[None] + embedded_timestep[:,
            None]).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)
    if self.adaln_single is None:
        height = width = int(hidden_states.shape[1] ** 0.5)
    hidden_states = hidden_states.reshape(shape=(-1, height, width, self.
        patch_size, self.patch_size, self.out_channels))
    hidden_states = torch.einsum('nhwpqc->nchpwq', hidden_states)
    output = hidden_states.reshape(shape=(-1, self.out_channels, height *
        self.patch_size, width * self.patch_size))
    return output
