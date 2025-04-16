def __init__(self, prior_tokenizer: CLIPTokenizer, prior_text_encoder:
    CLIPTextModelWithProjection, prior: PriorTransformer, prior_scheduler:
    KarrasDiffusionSchedulers, image_normalizer:
    StableUnCLIPImageNormalizer, image_noising_scheduler:
    KarrasDiffusionSchedulers, tokenizer: CLIPTokenizer, text_encoder:
    CLIPTextModelWithProjection, unet: UNet2DConditionModel, scheduler:
    KarrasDiffusionSchedulers, vae: AutoencoderKL):
    super().__init__()
    self.register_modules(prior_tokenizer=prior_tokenizer,
        prior_text_encoder=prior_text_encoder, prior=prior, prior_scheduler
        =prior_scheduler, image_normalizer=image_normalizer,
        image_noising_scheduler=image_noising_scheduler, tokenizer=
        tokenizer, text_encoder=text_encoder, unet=unet, scheduler=
        scheduler, vae=vae)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor)
