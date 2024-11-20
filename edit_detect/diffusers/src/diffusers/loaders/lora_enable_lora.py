def enable_lora(self):
    if not USE_PEFT_BACKEND:
        raise ValueError('PEFT backend is required for this method.')
    unet = getattr(self, self.unet_name) if not hasattr(self, 'unet'
        ) else self.unet
    unet.enable_lora()
    if hasattr(self, 'text_encoder'):
        self.enable_lora_for_text_encoder(self.text_encoder)
    if hasattr(self, 'text_encoder_2'):
        self.enable_lora_for_text_encoder(self.text_encoder_2)
