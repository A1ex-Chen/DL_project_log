def __init__(self, tokenizer: CLIPTokenizer, text_encoder:
    CLIPTextModelWithProjection, image_unet: UNet2DConditionModel,
    text_unet: UNetFlatConditionModel, vae: AutoencoderKL, scheduler:
    KarrasDiffusionSchedulers):
    super().__init__()
    self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder,
        image_unet=image_unet, text_unet=text_unet, vae=vae, scheduler=
        scheduler)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    if self.text_unet is not None:
        self._swap_unet_attention_blocks()
