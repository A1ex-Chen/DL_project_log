def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float,
    int], encoder_hidden_states: torch.Tensor, added_time_ids: torch.Tensor,
    return_dict: bool=True) ->Union[UNetSpatioTemporalConditionOutput, Tuple]:
    """
        The [`UNetSpatioTemporalConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.Tensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] instead
                of a plain tuple.
        Returns:
            [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] is
                returned, otherwise a `tuple` is returned where the first element is the sample tensor.
        """
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        is_mps = sample.device.type == 'mps'
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device
            )
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)
    batch_size, num_frames = sample.shape[:2]
    timesteps = timesteps.expand(batch_size)
    t_emb = self.time_proj(timesteps)
    t_emb = t_emb.to(dtype=sample.dtype)
    emb = self.time_embedding(t_emb)
    time_embeds = self.add_time_proj(added_time_ids.flatten())
    time_embeds = time_embeds.reshape((batch_size, -1))
    time_embeds = time_embeds.to(emb.dtype)
    aug_emb = self.add_embedding(time_embeds)
    emb = emb + aug_emb
    sample = sample.flatten(0, 1)
    emb = emb.repeat_interleave(num_frames, dim=0)
    encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames,
        dim=0)
    sample = self.conv_in(sample)
    image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample
        .dtype, device=sample.device)
    down_block_res_samples = sample,
    for downsample_block in self.down_blocks:
        if hasattr(downsample_block, 'has_cross_attention'
            ) and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(hidden_states=sample,
                temb=emb, encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator)
        else:
            sample, res_samples = downsample_block(hidden_states=sample,
                temb=emb, image_only_indicator=image_only_indicator)
        down_block_res_samples += res_samples
    sample = self.mid_block(hidden_states=sample, temb=emb,
        encoder_hidden_states=encoder_hidden_states, image_only_indicator=
        image_only_indicator)
    for i, upsample_block in enumerate(self.up_blocks):
        res_samples = down_block_res_samples[-len(upsample_block.resnets):]
        down_block_res_samples = down_block_res_samples[:-len(
            upsample_block.resnets)]
        if hasattr(upsample_block, 'has_cross_attention'
            ) and upsample_block.has_cross_attention:
            sample = upsample_block(hidden_states=sample, temb=emb,
                res_hidden_states_tuple=res_samples, encoder_hidden_states=
                encoder_hidden_states, image_only_indicator=
                image_only_indicator)
        else:
            sample = upsample_block(hidden_states=sample, temb=emb,
                res_hidden_states_tuple=res_samples, image_only_indicator=
                image_only_indicator)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])
    if not return_dict:
        return sample,
    return UNetSpatioTemporalConditionOutput(sample=sample)
