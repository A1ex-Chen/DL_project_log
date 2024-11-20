@classmethod
def from_unet(cls, unet: UNet2DConditionModel,
    controlnet_conditioning_channel_order: str='rgb',
    conditioning_embedding_out_channels: Optional[Tuple[int, ...]]=(16, 32,
    96, 256), load_weights_from_unet: bool=True, conditioning_channels: int=3):
    """
        Instantiate a [`ControlNetModel`] from [`UNet2DConditionModel`].

        Parameters:
            unet (`UNet2DConditionModel`):
                The UNet model weights to copy to the [`ControlNetModel`]. All configuration options are also copied
                where applicable.
        """
    transformer_layers_per_block = (unet.config.
        transformer_layers_per_block if 'transformer_layers_per_block' in
        unet.config else 1)
    encoder_hid_dim = (unet.config.encoder_hid_dim if 'encoder_hid_dim' in
        unet.config else None)
    encoder_hid_dim_type = (unet.config.encoder_hid_dim_type if 
        'encoder_hid_dim_type' in unet.config else None)
    addition_embed_type = (unet.config.addition_embed_type if 
        'addition_embed_type' in unet.config else None)
    addition_time_embed_dim = (unet.config.addition_time_embed_dim if 
        'addition_time_embed_dim' in unet.config else None)
    controlnet = cls(encoder_hid_dim=encoder_hid_dim, encoder_hid_dim_type=
        encoder_hid_dim_type, addition_embed_type=addition_embed_type,
        addition_time_embed_dim=addition_time_embed_dim,
        transformer_layers_per_block=transformer_layers_per_block,
        in_channels=unet.config.in_channels, flip_sin_to_cos=unet.config.
        flip_sin_to_cos, freq_shift=unet.config.freq_shift,
        down_block_types=unet.config.down_block_types, only_cross_attention
        =unet.config.only_cross_attention, block_out_channels=unet.config.
        block_out_channels, layers_per_block=unet.config.layers_per_block,
        downsample_padding=unet.config.downsample_padding,
        mid_block_scale_factor=unet.config.mid_block_scale_factor, act_fn=
        unet.config.act_fn, norm_num_groups=unet.config.norm_num_groups,
        norm_eps=unet.config.norm_eps, cross_attention_dim=unet.config.
        cross_attention_dim, attention_head_dim=unet.config.
        attention_head_dim, num_attention_heads=unet.config.
        num_attention_heads, use_linear_projection=unet.config.
        use_linear_projection, class_embed_type=unet.config.
        class_embed_type, num_class_embeds=unet.config.num_class_embeds,
        upcast_attention=unet.config.upcast_attention,
        resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
        projection_class_embeddings_input_dim=unet.config.
        projection_class_embeddings_input_dim, mid_block_type=unet.config.
        mid_block_type, controlnet_conditioning_channel_order=
        controlnet_conditioning_channel_order,
        conditioning_embedding_out_channels=
        conditioning_embedding_out_channels, conditioning_channels=
        conditioning_channels)
    if load_weights_from_unet:
        controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
        controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
        controlnet.time_embedding.load_state_dict(unet.time_embedding.
            state_dict())
        if controlnet.class_embedding:
            controlnet.class_embedding.load_state_dict(unet.class_embedding
                .state_dict())
        if hasattr(controlnet, 'add_embedding'):
            controlnet.add_embedding.load_state_dict(unet.add_embedding.
                state_dict())
        controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
        controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())
    return controlnet
