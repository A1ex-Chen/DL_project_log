def __init__(self, tokenizer: T5Tokenizer, text_encoder: T5EncoderModel,
    vae: AutoencoderKL, transformer: Transformer2DModel, scheduler:
    DPMSolverMultistepScheduler):
    super().__init__()
    self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder,
        vae=vae, transformer=transformer, scheduler=scheduler)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = PixArtImageProcessor(vae_scale_factor=self.
        vae_scale_factor)
