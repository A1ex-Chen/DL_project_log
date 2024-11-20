def __call__(self, sample: jnp.ndarray, timesteps: Union[jnp.ndarray, float,
    int], encoder_hidden_states: jnp.ndarray, added_cond_kwargs: Optional[
    Union[Dict, FrozenDict]]=None, down_block_additional_residuals:
    Optional[Tuple[jnp.ndarray, ...]]=None, mid_block_additional_residual:
    Optional[jnp.ndarray]=None, return_dict: bool=True, train: bool=False
    ) ->Union[FlaxUNet2DConditionOutput, Tuple[jnp.ndarray]]:
    """
        Args:
            sample (`jnp.ndarray`): (batch, channel, height, width) noisy inputs tensor
            timestep (`jnp.ndarray` or `float` or `int`): timesteps
            encoder_hidden_states (`jnp.ndarray`): (batch_size, sequence_length, hidden_size) encoder hidden states
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unets.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] instead of
                a plain tuple.
            train (`bool`, *optional*, defaults to `False`):
                Use deterministic functions and disable dropout when not training.

        Returns:
            [`~models.unets.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] or `tuple`:
            [`~models.unets.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.
        """
    if not isinstance(timesteps, jnp.ndarray):
        timesteps = jnp.array([timesteps], dtype=jnp.int32)
    elif isinstance(timesteps, jnp.ndarray) and len(timesteps.shape) == 0:
        timesteps = timesteps.astype(dtype=jnp.float32)
        timesteps = jnp.expand_dims(timesteps, 0)
    t_emb = self.time_proj(timesteps)
    t_emb = self.time_embedding(t_emb)
    aug_emb = None
    if self.addition_embed_type == 'text_time':
        if added_cond_kwargs is None:
            raise ValueError(
                f'Need to provide argument `added_cond_kwargs` for {self.__class__} when using `addition_embed_type={self.addition_embed_type}`'
                )
        text_embeds = added_cond_kwargs.get('text_embeds')
        if text_embeds is None:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
        time_ids = added_cond_kwargs.get('time_ids')
        if time_ids is None:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
        time_embeds = self.add_time_proj(jnp.ravel(time_ids))
        time_embeds = jnp.reshape(time_embeds, (text_embeds.shape[0], -1))
        add_embeds = jnp.concatenate([text_embeds, time_embeds], axis=-1)
        aug_emb = self.add_embedding(add_embeds)
    t_emb = t_emb + aug_emb if aug_emb is not None else t_emb
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
    if self.mid_block is not None:
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
