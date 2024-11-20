def __call__(self, sample, timesteps, encoder_hidden_states,
    down_block_additional_residuals=None, mid_block_additional_residual=
    None, return_dict: bool=True, train: bool=False) ->Union[
    FlaxUNet2DConditionOutput, Tuple]:
    """
        Args:
            sample (`jnp.ndarray`): (batch, channel, height, width) noisy inputs tensor
            timestep (`jnp.ndarray` or `float` or `int`): timesteps
            encoder_hidden_states (`jnp.ndarray`): (batch_size, sequence_length, hidden_size) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] instead of a
                plain tuple.
            train (`bool`, *optional*, defaults to `False`):
                Use deterministic functions and disable dropout when not training.

        Returns:
            [`~models.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is the sample tensor.
        """
    if not isinstance(timesteps, jnp.ndarray):
        timesteps = jnp.array([timesteps], dtype=jnp.int32)
    elif isinstance(timesteps, jnp.ndarray) and len(timesteps.shape) == 0:
        timesteps = timesteps.astype(dtype=jnp.float32)
        timesteps = jnp.expand_dims(timesteps, 0)
    t_emb = self.time_proj(timesteps)
    t_emb = self.time_embedding(t_emb)
    sample = jnp.transpose(sample, (0, 2, 3, 1))
    sample = self.conv_in(sample)
    down_block_res_samples = sample,
    for down_block in self.down_blocks:
        if isinstance(down_block, FlaxCrossAttnDownBlock2D):
            sample, res_samples = down_block(sample, t_emb,
                encoder_hidden_states, deterministic=not train)
        else:
            sample, res_samples = down_block(sample, t_emb, deterministic=
                not train)
        down_block_res_samples += res_samples
    if down_block_additional_residuals is not None:
        new_down_block_res_samples = ()
        for down_block_res_sample, down_block_additional_residual in zip(
            down_block_res_samples, down_block_additional_residuals):
            down_block_res_sample += down_block_additional_residual
            new_down_block_res_samples += down_block_res_sample,
        down_block_res_samples = new_down_block_res_samples
    sample = self.mid_block(sample, t_emb, encoder_hidden_states,
        deterministic=not train)
    if mid_block_additional_residual is not None:
        sample += mid_block_additional_residual
    for up_block in self.up_blocks:
        res_samples = down_block_res_samples[-(self.layers_per_block + 1):]
        down_block_res_samples = down_block_res_samples[:-(self.
            layers_per_block + 1)]
        if isinstance(up_block, FlaxCrossAttnUpBlock2D):
            sample = up_block(sample, temb=t_emb, encoder_hidden_states=
                encoder_hidden_states, res_hidden_states_tuple=res_samples,
                deterministic=not train)
        else:
            sample = up_block(sample, temb=t_emb, res_hidden_states_tuple=
                res_samples, deterministic=not train)
    sample = self.conv_norm_out(sample)
    sample = nn.silu(sample)
    sample = self.conv_out(sample)
    sample = jnp.transpose(sample, (0, 3, 1, 2))
    if not return_dict:
        return sample,
    return FlaxUNet2DConditionOutput(sample=sample)
