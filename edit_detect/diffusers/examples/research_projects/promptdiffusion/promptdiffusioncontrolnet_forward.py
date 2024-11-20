def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float,
    int], encoder_hidden_states: torch.Tensor, controlnet_cond: torch.
    Tensor, controlnet_query_cond: torch.Tensor, conditioning_scale: float=
    1.0, class_labels: Optional[torch.Tensor]=None, timestep_cond: Optional
    [torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]]=None,
    cross_attention_kwargs: Optional[Dict[str, Any]]=None, guess_mode: bool
    =False, return_dict: bool=True) ->Union[ControlNetOutput, Tuple[Tuple[
    torch.Tensor, ...], torch.Tensor]]:
    """
        The [`~PromptDiffusionControlNetModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor.
            timestep (`Union[torch.Tensor, float, int]`):
                The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states.
            controlnet_cond (`torch.Tensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            controlnet_query_cond (`torch.Tensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            conditioning_scale (`float`, defaults to `1.0`):
                The scale factor for ControlNet outputs.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond (`torch.Tensor`, *optional*, defaults to `None`):
                Additional conditional embeddings for timestep. If provided, the embeddings will be summed with the
                timestep_embedding passed through the `self.time_embedding` layer to obtain the final timestep
                embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            added_cond_kwargs (`dict`):
                Additional conditions for the Stable Diffusion XL UNet.
            cross_attention_kwargs (`dict[str]`, *optional*, defaults to `None`):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor`.
            guess_mode (`bool`, defaults to `False`):
                In this mode, the ControlNet encoder tries its best to recognize the input content of the input even if
                you remove all prompts. A `guidance_scale` between 3.0 and 5.0 is recommended.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~models.controlnet.ControlNetOutput`] instead of a plain tuple.

        Returns:
            [`~models.controlnet.ControlNetOutput`] **or** `tuple`:
                If `return_dict` is `True`, a [`~models.controlnet.ControlNetOutput`] is returned, otherwise a tuple is
                returned where the first element is the sample tensor.
        """
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
    t_emb = t_emb.to(dtype=sample.dtype)
    emb = self.time_embedding(t_emb, timestep_cond)
    aug_emb = None
    if self.class_embedding is not None:
        if class_labels is None:
            raise ValueError(
                'class_labels should be provided when num_class_embeds > 0')
        if self.config.class_embed_type == 'timestep':
            class_labels = self.time_proj(class_labels)
        class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
        emb = emb + class_emb
    if self.config.addition_embed_type is not None:
        if self.config.addition_embed_type == 'text':
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == 'text_time':
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
    emb = emb + aug_emb if aug_emb is not None else emb
    sample = self.conv_in(sample)
    controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
    controlnet_query_cond = self.controlnet_query_cond_embedding(
        controlnet_query_cond)
    sample = sample + controlnet_cond + controlnet_query_cond
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
        if hasattr(self.mid_block, 'has_cross_attention'
            ) and self.mid_block.has_cross_attention:
            sample = self.mid_block(sample, emb, encoder_hidden_states=
                encoder_hidden_states, attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs)
        else:
            sample = self.mid_block(sample, emb)
    controlnet_down_block_res_samples = ()
    for down_block_res_sample, controlnet_block in zip(down_block_res_samples,
        self.controlnet_down_blocks):
        down_block_res_sample = controlnet_block(down_block_res_sample)
        controlnet_down_block_res_samples = (
            controlnet_down_block_res_samples + (down_block_res_sample,))
    down_block_res_samples = controlnet_down_block_res_samples
    mid_block_res_sample = self.controlnet_mid_block(sample)
    if guess_mode and not self.config.global_pool_conditions:
        scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1,
            device=sample.device)
        scales = scales * conditioning_scale
        down_block_res_samples = [(sample * scale) for sample, scale in zip
            (down_block_res_samples, scales)]
        mid_block_res_sample = mid_block_res_sample * scales[-1]
    else:
        down_block_res_samples = [(sample * conditioning_scale) for sample in
            down_block_res_samples]
        mid_block_res_sample = mid_block_res_sample * conditioning_scale
    if self.config.global_pool_conditions:
        down_block_res_samples = [torch.mean(sample, dim=(2, 3), keepdim=
            True) for sample in down_block_res_samples]
        mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3),
            keepdim=True)
    if not return_dict:
        return down_block_res_samples, mid_block_res_sample
    return ControlNetOutput(down_block_res_samples=down_block_res_samples,
        mid_block_res_sample=mid_block_res_sample)
