def __init__(self, vae: AutoencoderKLTemporalDecoder, image_encoder:
    CLIPVisionModelWithProjection, unet: UNetSpatioTemporalConditionModel,
    scheduler: EulerDiscreteScheduler, feature_extractor: CLIPImageProcessor):
    super().__init__()
    self.register_modules(vae=vae, image_encoder=image_encoder, unet=unet,
        scheduler=scheduler, feature_extractor=feature_extractor)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.video_processor = VideoProcessor(do_resize=False, vae_scale_factor
        =self.vae_scale_factor)
