def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float,
    int], fps: torch.Tensor, image_latents: torch.Tensor, image_embeddings:
    Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.
    Tensor]=None, timestep_cond: Optional[torch.Tensor]=None,
    cross_attention_kwargs: Optional[Dict[str, Any]]=None, return_dict:
    bool=True) ->Union[UNet3DConditionOutput, Tuple[torch.Tensor]]:
    """
        The [`I2VGenXLUNet`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            fps (`torch.Tensor`): Frames per second for the video being generated. Used as a "micro-condition".
            image_latents (`torch.Tensor`): Image encodings from the VAE.
            image_embeddings (`torch.Tensor`):
                Projection embeddings of the conditioning image computed with a vision encoder.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_3d_condition.UNet3DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unet_3d_condition.UNet3DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_3d_condition.UNet3DConditionOutput`] is returned, otherwise
                a `tuple` is returned where the first element is the sample tensor.
        """
    batch_size, channels, num_frames, height, width = sample.shape
    default_overall_up_factor = 2 ** self.num_upsamplers
    forward_upsample_size = False
    upsample_size = None
    if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
        logger.info('Forward upsample size to force interpolation output size.'
            )
        forward_upsample_size = True
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        is_mps = sample.device.type == 'mps'
        if isinstance(timesteps, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device
            )
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = self.time_proj(timesteps)
    t_emb = t_emb.to(dtype=self.dtype)
    t_emb = self.time_embedding(t_emb, timestep_cond)
    fps = fps.expand(fps.shape[0])
    fps_emb = self.fps_embedding(self.time_proj(fps).to(dtype=self.dtype))
    emb = t_emb + fps_emb
    emb = emb.repeat_interleave(repeats=num_frames, dim=0)
    context_emb = sample.new_zeros(batch_size, 0, self.config.
        cross_attention_dim)
    context_emb = torch.cat([context_emb, encoder_hidden_states], dim=1)
    image_latents_for_context_embds = image_latents[:, :, :1, :]
    image_latents_context_embs = image_latents_for_context_embds.permute(0,
        2, 1, 3, 4).reshape(image_latents_for_context_embds.shape[0] *
        image_latents_for_context_embds.shape[2],
        image_latents_for_context_embds.shape[1],
        image_latents_for_context_embds.shape[3],
        image_latents_for_context_embds.shape[4])
    image_latents_context_embs = self.image_latents_context_embedding(
        image_latents_context_embs)
    _batch_size, _channels, _height, _width = image_latents_context_embs.shape
    image_latents_context_embs = image_latents_context_embs.permute(0, 2, 3, 1
        ).reshape(_batch_size, _height * _width, _channels)
    context_emb = torch.cat([context_emb, image_latents_context_embs], dim=1)
    image_emb = self.context_embedding(image_embeddings)
    image_emb = image_emb.view(-1, self.config.in_channels, self.config.
        cross_attention_dim)
    context_emb = torch.cat([context_emb, image_emb], dim=1)
    context_emb = context_emb.repeat_interleave(repeats=num_frames, dim=0)
    image_latents = image_latents.permute(0, 2, 1, 3, 4).reshape(
        image_latents.shape[0] * image_latents.shape[2], image_latents.
        shape[1], image_latents.shape[3], image_latents.shape[4])
    image_latents = self.image_latents_proj_in(image_latents)
    image_latents = image_latents[None, :].reshape(batch_size, num_frames,
        channels, height, width).permute(0, 3, 4, 1, 2).reshape(batch_size *
        height * width, num_frames, channels)
    image_latents = self.image_latents_temporal_encoder(image_latents)
    image_latents = image_latents.reshape(batch_size, height, width,
        num_frames, channels).permute(0, 4, 3, 1, 2)
    sample = torch.cat([sample, image_latents], dim=1)
    sample = sample.permute(0, 2, 1, 3, 4).reshape((sample.shape[0] *
        num_frames, -1) + sample.shape[3:])
    sample = self.conv_in(sample)
    sample = self.transformer_in(sample, num_frames=num_frames,
        cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]
    down_block_res_samples = sample,
    for downsample_block in self.down_blocks:
        if hasattr(downsample_block, 'has_cross_attention'
            ) and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(hidden_states=sample,
                temb=emb, encoder_hidden_states=context_emb, num_frames=
                num_frames, cross_attention_kwargs=cross_attention_kwargs)
        else:
            sample, res_samples = downsample_block(hidden_states=sample,
                temb=emb, num_frames=num_frames)
        down_block_res_samples += res_samples
    if self.mid_block is not None:
        sample = self.mid_block(sample, emb, encoder_hidden_states=
            context_emb, num_frames=num_frames, cross_attention_kwargs=
            cross_attention_kwargs)
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
                context_emb, upsample_size=upsample_size, num_frames=
                num_frames, cross_attention_kwargs=cross_attention_kwargs)
        else:
            sample = upsample_block(hidden_states=sample, temb=emb,
                res_hidden_states_tuple=res_samples, upsample_size=
                upsample_size, num_frames=num_frames)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    sample = sample[None, :].reshape((-1, num_frames) + sample.shape[1:]
        ).permute(0, 2, 1, 3, 4)
    if not return_dict:
        return sample,
    return UNet3DConditionOutput(sample=sample)
