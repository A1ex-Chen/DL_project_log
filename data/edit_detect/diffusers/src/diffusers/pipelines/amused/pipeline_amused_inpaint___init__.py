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
    self.mask_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor, do_normalize=False, do_binarize=True,
        do_convert_grayscale=True, do_resize=True)
    self.scheduler.register_to_config(masking_schedule='linear')
