def __call__(self, sample: jnp.ndarray, timesteps: Union[jnp.ndarray, float,
    int], encoder_hidden_states: jnp.ndarray, controlnet_cond: jnp.ndarray,
    conditioning_scale: float=1.0, return_dict: bool=True, train: bool=False
    ) ->Union[FlaxControlNetOutput, Tuple[Tuple[jnp.ndarray, ...], jnp.ndarray]
    ]:
    """
        Args:
            sample (`jnp.ndarray`): (batch, channel, height, width) noisy inputs tensor
            timestep (`jnp.ndarray` or `float` or `int`): timesteps
            encoder_hidden_states (`jnp.ndarray`): (batch_size, sequence_length, hidden_size) encoder hidden states
            controlnet_cond (`jnp.ndarray`): (batch, channel, height, width) the conditional input tensor
            conditioning_scale (`float`, *optional*, defaults to `1.0`): the scale factor for controlnet outputs
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unets.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] instead of
                a plain tuple.
            train (`bool`, *optional*, defaults to `False`):
                Use deterministic functions and disable dropout when not training.

        Returns:
            [`~models.unets.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] or `tuple`:
                [`~models.unets.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] if `return_dict` is True, otherwise
                a `tuple`. When returning a tuple, the first element is the sample tensor.
        """
    channel_order = self.controlnet_conditioning_channel_order
    if channel_order == 'bgr':
        controlnet_cond = jnp.flip(controlnet_cond, axis=1)
    if not isinstance(timesteps, jnp.ndarray):
        timesteps = jnp.array([timesteps], dtype=jnp.int32)
    elif isinstance(timesteps, jnp.ndarray) and len(timesteps.shape) == 0:
        timesteps = timesteps.astype(dtype=jnp.float32)
        timesteps = jnp.expand_dims(timesteps, 0)
    t_emb = self.time_proj(timesteps)
    t_emb = self.time_embedding(t_emb)
    sample = jnp.transpose(sample, (0, 2, 3, 1))
    sample = self.conv_in(sample)
    controlnet_cond = jnp.transpose(controlnet_cond, (0, 2, 3, 1))
    controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
    sample += controlnet_cond
    down_block_res_samples = sample,
    for down_block in self.down_blocks:
        if isinstance(down_block, FlaxCrossAttnDownBlock2D):
            sample, res_samples = down_block(sample, t_emb,
                encoder_hidden_states, deterministic=not train)
        else:
            sample, res_samples = down_block(sample, t_emb, deterministic=
                not train)
        down_block_res_samples += res_samples
    sample = self.mid_block(sample, t_emb, encoder_hidden_states,
        deterministic=not train)
    controlnet_down_block_res_samples = ()
    for down_block_res_sample, controlnet_block in zip(down_block_res_samples,
        self.controlnet_down_blocks):
        down_block_res_sample = controlnet_block(down_block_res_sample)
        controlnet_down_block_res_samples += down_block_res_sample,
    down_block_res_samples = controlnet_down_block_res_samples
    mid_block_res_sample = self.controlnet_mid_block(sample)
    down_block_res_samples = [(sample * conditioning_scale) for sample in
        down_block_res_samples]
    mid_block_res_sample *= conditioning_scale
    if not return_dict:
        return down_block_res_samples, mid_block_res_sample
    return FlaxControlNetOutput(down_block_res_samples=
        down_block_res_samples, mid_block_res_sample=mid_block_res_sample)
