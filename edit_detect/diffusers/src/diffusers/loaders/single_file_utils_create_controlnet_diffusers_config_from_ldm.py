def create_controlnet_diffusers_config_from_ldm(original_config, checkpoint,
    image_size=None, **kwargs):
    if image_size is not None:
        deprecation_message = (
            'Configuring ControlNetModel with the `image_size` argumentis deprecated and will be ignored in future versions.'
            )
        deprecate('image_size', '1.0.0', deprecation_message)
    image_size = set_image_size(checkpoint, image_size=image_size)
    unet_params = original_config['model']['params']['control_stage_config'][
        'params']
    diffusers_unet_config = create_unet_diffusers_config_from_ldm(
        original_config, image_size=image_size)
    controlnet_config = {'conditioning_channels': unet_params[
        'hint_channels'], 'in_channels': diffusers_unet_config[
        'in_channels'], 'down_block_types': diffusers_unet_config[
        'down_block_types'], 'block_out_channels': diffusers_unet_config[
        'block_out_channels'], 'layers_per_block': diffusers_unet_config[
        'layers_per_block'], 'cross_attention_dim': diffusers_unet_config[
        'cross_attention_dim'], 'attention_head_dim': diffusers_unet_config
        ['attention_head_dim'], 'use_linear_projection':
        diffusers_unet_config['use_linear_projection'], 'class_embed_type':
        diffusers_unet_config['class_embed_type'], 'addition_embed_type':
        diffusers_unet_config['addition_embed_type'],
        'addition_time_embed_dim': diffusers_unet_config[
        'addition_time_embed_dim'], 'projection_class_embeddings_input_dim':
        diffusers_unet_config['projection_class_embeddings_input_dim'],
        'transformer_layers_per_block': diffusers_unet_config[
        'transformer_layers_per_block']}
    return controlnet_config
