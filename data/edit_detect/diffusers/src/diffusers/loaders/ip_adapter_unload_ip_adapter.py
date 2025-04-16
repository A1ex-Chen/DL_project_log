def unload_ip_adapter(self):
    """
        Unloads the IP Adapter weights

        Examples:

        ```python
        >>> # Assuming `pipeline` is already loaded with the IP Adapter weights.
        >>> pipeline.unload_ip_adapter()
        >>> ...
        ```
        """
    if hasattr(self, 'image_encoder') and getattr(self, 'image_encoder', None
        ) is not None:
        self.image_encoder = None
        self.register_to_config(image_encoder=[None, None])
    if not hasattr(self, 'safety_checker'):
        if hasattr(self, 'feature_extractor') and getattr(self,
            'feature_extractor', None) is not None:
            self.feature_extractor = None
            self.register_to_config(feature_extractor=[None, None])
    self.unet.encoder_hid_proj = None
    self.config.encoder_hid_dim_type = None
    attn_procs = {}
    for name, value in self.unet.attn_processors.items():
        attn_processor_class = AttnProcessor2_0() if hasattr(F,
            'scaled_dot_product_attention') else AttnProcessor()
        attn_procs[name] = attn_processor_class if isinstance(value, (
            IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)
            ) else value.__class__()
    self.unet.set_attn_processor(attn_procs)
