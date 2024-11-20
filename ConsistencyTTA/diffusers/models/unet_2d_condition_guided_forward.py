def forward(self, sample: torch.FloatTensor, timestep: Union[torch.Tensor,
    float, int], guidance: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor, class_labels: Optional[torch.
    Tensor]=None, timestep_cond: Optional[torch.Tensor]=None, guidance_cond:
    Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=
    None, cross_attention_kwargs: Optional[Dict[str, Any]]=None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]]=None,
    down_block_additional_residuals: Optional[Tuple[torch.Tensor]]=None,
    mid_block_additional_residual: Optional[torch.Tensor]=None,
    encoder_attention_mask: Optional[torch.Tensor]=None, return_dict: bool=
    True, **kwargs) ->Union[UNet2DConditionOutput, Tuple]:
    """
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): 
                (batch, sequence_length, feature_dim) encoder hidden states
            encoder_attention_mask (`torch.Tensor`):
                (batch, sequence_length) cross-attention mask, applied to encoder_hidden_states. True = keep,
                False = discard. Mask will be converted into a bias, which adds large negative values to
                attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`]
                instead of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined
                under `self.processor` in [diffusers.cross_attention]
                (https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            added_cond_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified includes additonal conditions that can be used for
                additonal time embeddings or encoder hidden states projections. See the configurations
                `encoder_hid_dim_type` and `addition_embed_type` for more information.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`.
                When returning a tuple, the first element is the sample tensor.
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
    if encoder_attention_mask is not None:
        encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)
            ) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0
    timestep = self._prepare_tensor(timestep, sample.device).expand(sample.
        shape[0])
    t_emb = self.time_proj(timestep).to(dtype=sample.dtype)
    t_emb = self.time_embedding(t_emb, timestep_cond)
    guidance = self._prepare_tensor(guidance, sample.device).expand(sample.
        shape[0])
    g_emb = self.guidance_proj(guidance).to(dtype=sample.dtype)
    g_emb = self.guidance_embedding(g_emb, guidance_cond)
    if self.class_embedding is None:
        emb = t_emb + g_emb
    else:
        if class_labels is None:
            raise ValueError(
                'class_labels should be provided when num_class_embeds > 0')
        if self.config.class_embed_type == 'timestep':
            class_labels = self.time_proj(class_labels).to(dtype=sample.dtype)
        class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
        if self.config.class_embeddings_concat:
            emb = torch.cat([t_emb, g_emb, class_emb], dim=-1)
        else:
            emb = t_emb + g_emb + class_emb
    if self.config.addition_embed_type == 'text':
        aug_emb = self.add_embedding(encoder_hidden_states)
        emb = emb + aug_emb
    elif self.config.addition_embed_type == 'text_image':
        if 'image_embeds' not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
        image_embs = added_cond_kwargs.get('image_embeds')
        text_embs = added_cond_kwargs.get('text_embeds', encoder_hidden_states)
        aug_emb = self.add_embedding(text_embs, image_embs)
        emb = emb + aug_emb
    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)
    if (self.encoder_hid_proj is not None and self.config.
        encoder_hid_dim_type == 'text_proj'):
        encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
    elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == 'text_image_proj':
        if 'image_embeds' not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
        image_embeds = added_cond_kwargs.get('image_embeds')
        encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states,
            image_embeds)
    sample = self.conv_in(sample)
    down_block_res_samples = sample,
    for downsample_block in self.down_blocks:
        if hasattr(downsample_block, 'has_cross_attention'
            ) and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(hidden_states=sample,
                temb=emb, encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask, cross_attention_kwargs=
                cross_attention_kwargs, encoder_attention_mask=
                encoder_attention_mask)
        else:
            sample, res_samples = downsample_block(hidden_states=sample,
                temb=emb)
        down_block_res_samples += res_samples
    if down_block_additional_residuals is not None:
        new_down_block_res_samples = ()
        for down_block_res_sample, down_block_additional_residual in zip(
            down_block_res_samples, down_block_additional_residuals):
            down_block_res_sample = (down_block_res_sample +
                down_block_additional_residual)
            new_down_block_res_samples = new_down_block_res_samples + (
                down_block_res_sample,)
        down_block_res_samples = new_down_block_res_samples
    if self.mid_block is not None:
        sample = self.mid_block(sample, emb, encoder_hidden_states=
            encoder_hidden_states, attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            encoder_attention_mask=encoder_attention_mask)
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
                encoder_hidden_states, cross_attention_kwargs=
                cross_attention_kwargs, upsample_size=upsample_size,
                attention_mask=attention_mask, encoder_attention_mask=
                encoder_attention_mask)
        else:
            sample = upsample_block(hidden_states=sample, temb=emb,
                res_hidden_states_tuple=res_samples, upsample_size=
                upsample_size)
    if self.conv_norm_out:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    if not return_dict:
        return sample,
    return UNet2DConditionOutput(sample=sample)
