def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float,
    int], encoder_hidden_states: torch.Tensor, timestep_cond: Optional[
    torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None,
    cross_attention_kwargs: Optional[Dict[str, Any]]=None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]]=None,
    down_block_additional_residuals: Optional[Tuple[torch.Tensor]]=None,
    mid_block_additional_residual: Optional[torch.Tensor]=None, return_dict:
    bool=True) ->Union[UNet3DConditionOutput, Tuple[torch.Tensor]]:
    """
        The [`UNetMotionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the timestep embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_3d_condition.UNet3DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unet_3d_condition.UNet3DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_3d_condition.UNet3DConditionOutput`] is returned, otherwise
                a `tuple` is returned where the first element is the sample tensor.
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
    aug_emb = None
    if self.config.addition_embed_type == 'text_time':
        if 'text_embeds' not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
        text_embeds = added_cond_kwargs.get('text_embeds')
        if 'time_ids' not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
        time_ids = added_cond_kwargs.get('time_ids')
        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(add_embeds)
    emb = emb if aug_emb is None else emb + aug_emb
    emb = emb.repeat_interleave(repeats=num_frames, dim=0)
    encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats
        =num_frames, dim=0)
    if (self.encoder_hid_proj is not None and self.config.
        encoder_hid_dim_type == 'ip_image_proj'):
        if 'image_embeds' not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
        image_embeds = added_cond_kwargs.get('image_embeds')
        image_embeds = self.encoder_hid_proj(image_embeds)
        image_embeds = [image_embed.repeat_interleave(repeats=num_frames,
            dim=0) for image_embed in image_embeds]
        encoder_hidden_states = encoder_hidden_states, image_embeds
    sample = sample.permute(0, 2, 1, 3, 4).reshape((sample.shape[0] *
        num_frames, -1) + sample.shape[3:])
    sample = self.conv_in(sample)
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
        if hasattr(self.mid_block, 'motion_modules'):
            sample = self.mid_block(sample, emb, encoder_hidden_states=
                encoder_hidden_states, attention_mask=attention_mask,
                num_frames=num_frames, cross_attention_kwargs=
                cross_attention_kwargs)
        else:
            sample = self.mid_block(sample, emb, encoder_hidden_states=
                encoder_hidden_states, attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs)
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
