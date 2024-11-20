def forward(self, sample: torch.FloatTensor, timestep: Union[torch.Tensor,
    float, int], encoder_hidden_states: torch.Tensor, class_labels:
    Optional[torch.Tensor]=None, timestep_cond: Optional[torch.Tensor]=None,
    attention_mask: Optional[torch.Tensor]=None, cross_attention_kwargs:
    Optional[Dict[str, Any]]=None, down_block_additional_residuals:
    Optional[Tuple[torch.Tensor]]=None, mid_block_additional_residual:
    Optional[torch.Tensor]=None, return_dict: bool=True) ->Union[
    UNet3DConditionOutput, Tuple]:
    """
        Args:
            sample (`torch.FloatTensor`): (batch, num_frames, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet3DConditionOutput`] instead of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Returns:
            [`~models.unet_2d_condition.UNet3DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet3DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
    default_overall_up_factor = 2 ** self.num_upsamplers
    forward_upsample_size = False
    upsample_size = None
    if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
        logger.info('Forward upsample size to force interpolation output size.'
            )
        forward_upsample_size = True
    if attention_mask is not None:
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)
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
    num_frames = sample.shape[2]
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = self.time_proj(timesteps)
    t_emb = t_emb.to(dtype=self.dtype)
    emb = self.time_embedding(t_emb, timestep_cond)
    emb = emb.repeat_interleave(repeats=num_frames, dim=0)
    encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats
        =num_frames, dim=0)
    sample = sample.permute(0, 2, 1, 3, 4).reshape((sample.shape[0] *
        num_frames, -1) + sample.shape[3:])
    sample = self.conv_in(sample)
    sample = self.transformer_in(sample, num_frames=num_frames).sample
    down_block_res_samples = sample,
    for downsample_block in self.down_blocks:
        if hasattr(downsample_block, 'has_cross_attention'
            ) and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(hidden_states=sample,
                temb=emb, encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask, num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs)
        else:
            sample, res_samples = downsample_block(hidden_states=sample,
                temb=emb, num_frames=num_frames)
        down_block_res_samples += res_samples
    if down_block_additional_residuals is not None:
        new_down_block_res_samples = ()
        for down_block_res_sample, down_block_additional_residual in zip(
            down_block_res_samples, down_block_additional_residuals):
            down_block_res_sample = (down_block_res_sample +
                down_block_additional_residual)
            new_down_block_res_samples += down_block_res_sample,
        down_block_res_samples = new_down_block_res_samples
    if self.mid_block is not None:
        sample = self.mid_block(sample, emb, encoder_hidden_states=
            encoder_hidden_states, attention_mask=attention_mask,
            num_frames=num_frames, cross_attention_kwargs=
            cross_attention_kwargs)
    if mid_block_additional_residual is not None:
        sample = sample + mid_block_additional_residual
    for i, upsample_block in enumerate(self.up_blocks):
        is_final_block = i == len(self.up_blocks) - 1
        res_samples = down_block_res_samples[-len(upsample_block.resnets):]
        down_block_res_samples = down_block_res_samples[:-len(
            upsample_block.resnets)]
        if not is_final_block and forward_upsample_size:
            upsample_size = down_block_res_samples[-1].shape[2:]
        if hasattr(upsample_block, 'has_cross_attention'
            ) and upsample_block.has_cross_attention:
            sample = upsample_block(hidden_states=sample, temb=emb,
                res_hidden_states_tuple=res_samples, encoder_hidden_states=
                encoder_hidden_states, upsample_size=upsample_size,
                attention_mask=attention_mask, num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs)
        else:
            sample = upsample_block(hidden_states=sample, temb=emb,
                res_hidden_states_tuple=res_samples, upsample_size=
                upsample_size, num_frames=num_frames)
    if self.conv_norm_out:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    sample = sample[None, :].reshape((-1, num_frames) + sample.shape[1:]
        ).permute(0, 2, 1, 3, 4)
    if not return_dict:
        return sample,
    return UNet3DConditionOutput(sample=sample)
