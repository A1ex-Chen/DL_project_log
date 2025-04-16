def unfuse_lora(self, unfuse_unet: bool=True, unfuse_text_encoder: bool=True):
    """
        Reverses the effect of
        [`pipe.fuse_lora()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraLoaderMixin.fuse_lora).

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            unfuse_unet (`bool`, defaults to `True`): Whether to unfuse the UNet LoRA parameters.
            unfuse_text_encoder (`bool`, defaults to `True`):
                Whether to unfuse the text encoder LoRA parameters. If the text encoder wasn't monkey-patched with the
                LoRA parameters then it won't have any effect.
        """
    from peft.tuners.tuners_utils import BaseTunerLayer
    unet = getattr(self, self.unet_name) if not hasattr(self, 'unet'
        ) else self.unet
    if unfuse_unet:
        for module in unet.modules():
            if isinstance(module, BaseTunerLayer):
                module.unmerge()

    def unfuse_text_encoder_lora(text_encoder):
        for module in text_encoder.modules():
            if isinstance(module, BaseTunerLayer):
                module.unmerge()
    if unfuse_text_encoder:
        if hasattr(self, 'text_encoder'):
            unfuse_text_encoder_lora(self.text_encoder)
        if hasattr(self, 'text_encoder_2'):
            unfuse_text_encoder_lora(self.text_encoder_2)
    self.num_fused_loras -= 1
