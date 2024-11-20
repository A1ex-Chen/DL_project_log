def __init__(self, vqvae: VQModel, tokenizer: CLIPTokenizer, text_encoder:
    CLIPTextModelWithProjection, transformer: UVit2DModel, scheduler:
    AmusedScheduler):
    super().__init__()
    self.register_modules(vqvae=vqvae, tokenizer=tokenizer, text_encoder=
        text_encoder, transformer=transformer, scheduler=scheduler)
    self.vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1
        )
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor, do_normalize=False)
