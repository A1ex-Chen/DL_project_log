def set_ip_adapter_scale(self, scale):
    unet = getattr(self, self.unet_name) if not hasattr(self, 'unet'
        ) else self.unet
    for attn_processor in unet.attn_processors.values():
        if isinstance(attn_processor, (IPAdapterAttnProcessor,
            IPAdapterAttnProcessor2_0)):
            attn_processor.scale = [scale]
