def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, scheduler:
    DDIMScheduler, mixing_coeff: float=0.93, leapfrog_steps: bool=True):
    self.mixing_coeff = mixing_coeff
    self.leapfrog_steps = leapfrog_steps
    super().__init__()
    self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=
        tokenizer, unet=unet, scheduler=scheduler)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor)
