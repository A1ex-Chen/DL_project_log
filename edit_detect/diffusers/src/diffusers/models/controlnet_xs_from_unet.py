@classmethod
def from_unet(cls, unet: UNet2DConditionModel, controlnet: Optional[
    ControlNetXSAdapter]=None, size_ratio: Optional[float]=None,
    ctrl_block_out_channels: Optional[List[float]]=None, time_embedding_mix:
    Optional[float]=None, ctrl_optional_kwargs: Optional[Dict]=None):
    """
        Instantiate a [`UNetControlNetXSModel`] from a [`UNet2DConditionModel`] and an optional [`ControlNetXSAdapter`]
        .

        Parameters:
            unet (`UNet2DConditionModel`):
                The UNet model we want to control.
            controlnet (`ControlNetXSAdapter`):
                The ConntrolNet-XS adapter with which the UNet will be fused. If none is given, a new ConntrolNet-XS
                adapter will be created.
            size_ratio (float, *optional*, defaults to `None`):
                Used to contruct the controlnet if none is given. See [`ControlNetXSAdapter.from_unet`] for details.
            ctrl_block_out_channels (`List[int]`, *optional*, defaults to `None`):
                Used to contruct the controlnet if none is given. See [`ControlNetXSAdapter.from_unet`] for details,
                where this parameter is called `block_out_channels`.
            time_embedding_mix (`float`, *optional*, defaults to None):
                Used to contruct the controlnet if none is given. See [`ControlNetXSAdapter.from_unet`] for details.
            ctrl_optional_kwargs (`Dict`, *optional*, defaults to `None`):
                Passed to the `init` of the new controlent if no controlent was given.
        """
    if controlnet is None:
        controlnet = ControlNetXSAdapter.from_unet(unet, size_ratio,
            ctrl_block_out_channels, **ctrl_optional_kwargs)
    elif any(o is not None for o in (size_ratio, ctrl_block_out_channels,
        time_embedding_mix, ctrl_optional_kwargs)):
        raise ValueError(
            'When a controlnet is passed, none of these parameters should be passed: size_ratio, ctrl_block_out_channels, time_embedding_mix, ctrl_optional_kwargs.'
            )
    params_for_unet = ['sample_size', 'down_block_types', 'up_block_types',
        'block_out_channels', 'norm_num_groups', 'cross_attention_dim',
        'transformer_layers_per_block', 'addition_embed_type',
        'addition_time_embed_dim', 'upcast_attention', 'time_cond_proj_dim',
        'projection_class_embeddings_input_dim']
    params_for_unet = {k: v for k, v in unet.config.items() if k in
        params_for_unet}
    params_for_unet['num_attention_heads'] = unet.config.attention_head_dim
    params_for_controlnet = ['conditioning_channels',
        'conditioning_embedding_out_channels', 'conditioning_channel_order',
        'learn_time_embedding', 'block_out_channels', 'num_attention_heads',
        'max_norm_num_groups']
    params_for_controlnet = {('ctrl_' + k): v for k, v in controlnet.config
        .items() if k in params_for_controlnet}
    params_for_controlnet['time_embedding_mix'
        ] = controlnet.config.time_embedding_mix
    model = cls.from_config({**params_for_unet, **params_for_controlnet})
    modules_from_unet = ['time_embedding', 'conv_in', 'conv_norm_out',
        'conv_out']
    for m in modules_from_unet:
        getattr(model, 'base_' + m).load_state_dict(getattr(unet, m).
            state_dict())
    optional_modules_from_unet = ['add_time_proj', 'add_embedding']
    for m in optional_modules_from_unet:
        if hasattr(unet, m) and getattr(unet, m) is not None:
            getattr(model, 'base_' + m).load_state_dict(getattr(unet, m).
                state_dict())
    model.controlnet_cond_embedding.load_state_dict(controlnet.
        controlnet_cond_embedding.state_dict())
    model.ctrl_conv_in.load_state_dict(controlnet.conv_in.state_dict())
    if controlnet.time_embedding is not None:
        model.ctrl_time_embedding.load_state_dict(controlnet.time_embedding
            .state_dict())
    model.control_to_base_for_conv_in.load_state_dict(controlnet.
        control_to_base_for_conv_in.state_dict())
    model.down_blocks = nn.ModuleList(ControlNetXSCrossAttnDownBlock2D.
        from_modules(b, c) for b, c in zip(unet.down_blocks, controlnet.
        down_blocks))
    model.mid_block = ControlNetXSCrossAttnMidBlock2D.from_modules(unet.
        mid_block, controlnet.mid_block)
    model.up_blocks = nn.ModuleList(ControlNetXSCrossAttnUpBlock2D.
        from_modules(b, c) for b, c in zip(unet.up_blocks, controlnet.
        up_connections))
    model.to(unet.dtype)
    return model
