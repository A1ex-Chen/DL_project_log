def __call__(self, attn: Attention, hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor]=None, attention_mask:
    Optional[torch.Tensor]=None, temb: Optional[torch.Tensor]=None, scale:
    float=1.0, ip_adapter_masks: Optional[torch.Tensor]=None):
    residual = hidden_states
    if encoder_hidden_states is not None:
        if isinstance(encoder_hidden_states, tuple):
            encoder_hidden_states, ip_hidden_states = encoder_hidden_states
        else:
            deprecation_message = (
                'You have passed a tensor as `encoder_hidden_states`. This is deprecated and will be removed in a future release. Please make sure to update your script to pass `encoder_hidden_states` as a tuple to suppress this warning.'
                )
            deprecate('encoder_hidden_states not a tuple', '1.0.0',
                deprecation_message, standard_warn=False)
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
            encoder_hidden_states, ip_hidden_states = encoder_hidden_states[
                :, :end_pos, :], [encoder_hidden_states[:, end_pos:, :]]
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)
    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width
            ).transpose(1, 2)
    batch_size, sequence_length, _ = (hidden_states.shape if 
        encoder_hidden_states is None else encoder_hidden_states.shape)
    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask,
            sequence_length, batch_size)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1,
            attention_mask.shape[-1])
    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)
            ).transpose(1, 2)
    query = attn.to_q(hidden_states)
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(
            encoder_hidden_states)
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)
    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    hidden_states = F.scaled_dot_product_attention(query, key, value,
        attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, 
        attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)
    if ip_adapter_masks is not None:
        if not isinstance(ip_adapter_masks, List):
            ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
        if not len(ip_adapter_masks) == len(self.scale) == len(ip_hidden_states
            ):
            raise ValueError(
                f'Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match length of self.scale array ({len(self.scale)}) and number of ip_hidden_states ({len(ip_hidden_states)})'
                )
        else:
            for index, (mask, scale, ip_state) in enumerate(zip(
                ip_adapter_masks, self.scale, ip_hidden_states)):
                if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
                    raise ValueError(
                        'Each element of the ip_adapter_masks array should be a tensor with shape [1, num_images_for_ip_adapter, height, width]. Please use `IPAdapterMaskProcessor` to preprocess your mask'
                        )
                if mask.shape[1] != ip_state.shape[1]:
                    raise ValueError(
                        f'Number of masks ({mask.shape[1]}) does not match number of ip images ({ip_state.shape[1]}) at index {index}'
                        )
                if isinstance(scale, list) and not len(scale) == mask.shape[1]:
                    raise ValueError(
                        f'Number of masks ({mask.shape[1]}) does not match number of scales ({len(scale)}) at index {index}'
                        )
    else:
        ip_adapter_masks = [None] * len(self.scale)
    for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
        ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip,
        ip_adapter_masks):
        skip = False
        if isinstance(scale, list):
            if all(s == 0 for s in scale):
                skip = True
        elif scale == 0:
            skip = True
        if not skip:
            if mask is not None:
                if not isinstance(scale, list):
                    scale = [scale] * mask.shape[1]
                current_num_images = mask.shape[1]
                for i in range(current_num_images):
                    ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                    ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])
                    ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim
                        ).transpose(1, 2)
                    ip_value = ip_value.view(batch_size, -1, attn.heads,
                        head_dim).transpose(1, 2)
                    _current_ip_hidden_states = F.scaled_dot_product_attention(
                        query, ip_key, ip_value, attn_mask=None, dropout_p=
                        0.0, is_causal=False)
                    _current_ip_hidden_states = (_current_ip_hidden_states.
                        transpose(1, 2).reshape(batch_size, -1, attn.heads *
                        head_dim))
                    _current_ip_hidden_states = _current_ip_hidden_states.to(
                        query.dtype)
                    mask_downsample = IPAdapterMaskProcessor.downsample(mask
                        [:, i, :, :], batch_size, _current_ip_hidden_states
                        .shape[1], _current_ip_hidden_states.shape[2])
                    mask_downsample = mask_downsample.to(dtype=query.dtype,
                        device=query.device)
                    hidden_states = hidden_states + scale[i] * (
                        _current_ip_hidden_states * mask_downsample)
            else:
                ip_key = to_k_ip(current_ip_hidden_states)
                ip_value = to_v_ip(current_ip_hidden_states)
                ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim
                    ).transpose(1, 2)
                ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim
                    ).transpose(1, 2)
                current_ip_hidden_states = F.scaled_dot_product_attention(query
                    , ip_key, ip_value, attn_mask=None, dropout_p=0.0,
                    is_causal=False)
                current_ip_hidden_states = current_ip_hidden_states.transpose(
                    1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                current_ip_hidden_states = current_ip_hidden_states.to(query
                    .dtype)
                hidden_states = (hidden_states + scale *
                    current_ip_hidden_states)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size,
            channel, height, width)
    if attn.residual_connection:
        hidden_states = hidden_states + residual
    hidden_states = hidden_states / attn.rescale_output_factor
    return hidden_states