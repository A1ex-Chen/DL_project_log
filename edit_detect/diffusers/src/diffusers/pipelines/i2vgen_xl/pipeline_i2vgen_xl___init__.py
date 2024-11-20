def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer, image_encoder: CLIPVisionModelWithProjection,
    feature_extractor: CLIPImageProcessor, unet: I2VGenXLUNet, scheduler:
    DDIMScheduler):
    super().__init__()
    self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=
        tokenizer, image_encoder=image_encoder, feature_extractor=
        feature_extractor, unet=unet, scheduler=scheduler)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.video_processor = VideoProcessor(vae_scale_factor=self.
        vae_scale_factor, do_resize=False)
