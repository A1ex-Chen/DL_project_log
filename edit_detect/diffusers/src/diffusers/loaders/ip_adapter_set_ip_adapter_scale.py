def set_ip_adapter_scale(self, scale):
    """
        Set IP-Adapter scales per-transformer block. Input `scale` could be a single config or a list of configs for
        granular control over each IP-Adapter behavior. A config can be a float or a dictionary.

        Example:

        ```py
        # To use original IP-Adapter
        scale = 1.0
        pipeline.set_ip_adapter_scale(scale)

        # To use style block only
        scale = {
            "up": {"block_0": [0.0, 1.0, 0.0]},
        }
        pipeline.set_ip_adapter_scale(scale)

        # To use style+layout blocks
        scale = {
            "down": {"block_2": [0.0, 1.0]},
            "up": {"block_0": [0.0, 1.0, 0.0]},
        }
        pipeline.set_ip_adapter_scale(scale)

        # To use style and layout from 2 reference images
        scales = [{"down": {"block_2": [0.0, 1.0]}}, {"up": {"block_0": [0.0, 1.0, 0.0]}}]
        pipeline.set_ip_adapter_scale(scales)
        ```
        """
    unet = getattr(self, self.unet_name) if not hasattr(self, 'unet'
        ) else self.unet
    if not isinstance(scale, list):
        scale = [scale]
    scale_configs = _maybe_expand_lora_scales(unet, scale, default_scale=0.0)
    for attn_name, attn_processor in unet.attn_processors.items():
        if isinstance(attn_processor, (IPAdapterAttnProcessor,
            IPAdapterAttnProcessor2_0)):
            if len(scale_configs) != len(attn_processor.scale):
                raise ValueError(
                    f'Cannot assign {len(scale_configs)} scale_configs to {len(attn_processor.scale)} IP-Adapter.'
                    )
            elif len(scale_configs) == 1:
                scale_configs = scale_configs * len(attn_processor.scale)
            for i, scale_config in enumerate(scale_configs):
                if isinstance(scale_config, dict):
                    for k, s in scale_config.items():
                        if attn_name.startswith(k):
                            attn_processor.scale[i] = s
                else:
                    attn_processor.scale[i] = scale_config
