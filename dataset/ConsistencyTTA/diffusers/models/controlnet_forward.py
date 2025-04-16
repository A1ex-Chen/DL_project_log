def forward(self, sample: torch.FloatTensor, timestep: Union[torch.Tensor,
    float, int], encoder_hidden_states: torch.Tensor, controlnet_cond:
    torch.FloatTensor, conditioning_scale: float=1.0, class_labels:
    Optional[torch.Tensor]=None, timestep_cond: Optional[torch.Tensor]=None,
    attention_mask: Optional[torch.Tensor]=None, cross_attention_kwargs:
    Optional[Dict[str, Any]]=None, return_dict: bool=True) ->Union[
    ControlNetOutput, Tuple]:
    channel_order = self.config.controlnet_conditioning_channel_order
    if channel_order == 'rgb':
        ...
    elif channel_order == 'bgr':
        controlnet_cond = torch.flip(controlnet_cond, dims=[1])
    else:
        raise ValueError(
            f'unknown `controlnet_conditioning_channel_order`: {channel_order}'
            )
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
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = self.time_proj(timesteps)
    t_emb = t_emb.to(dtype=self.dtype)
    emb = self.time_embedding(t_emb, timestep_cond)
    if self.class_embedding is not None:
        if class_labels is None:
            raise ValueError(
                'class_labels should be provided when num_class_embeds > 0')
        if self.config.class_embed_type == 'timestep':
            class_labels = self.time_proj(class_labels)
        class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
        emb = emb + class_emb
    sample = self.conv_in(sample)
    controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
    sample += controlnet_cond
    down_block_res_samples = sample,
    for downsample_block in self.down_blocks:
        if hasattr(downsample_block, 'has_cross_attention'
            ) and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(hidden_states=sample,
                temb=emb, encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask, cross_attention_kwargs=
                cross_attention_kwargs)
        else:
            sample, res_samples = downsample_block(hidden_states=sample,
                temb=emb)
        down_block_res_samples += res_samples
    if self.mid_block is not None:
        sample = self.mid_block(sample, emb, encoder_hidden_states=
            encoder_hidden_states, attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs)
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
    return ControlNetOutput(down_block_res_samples=down_block_res_samples,
        mid_block_res_sample=mid_block_res_sample)
