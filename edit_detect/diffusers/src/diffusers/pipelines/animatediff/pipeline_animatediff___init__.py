def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer, unet: Union[UNet2DConditionModel,
    UNetMotionModel], motion_adapter: MotionAdapter, scheduler: Union[
    DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler,
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler], feature_extractor: CLIPImageProcessor=
    None, image_encoder: CLIPVisionModelWithProjection=None):
    super().__init__()
    if isinstance(unet, UNet2DConditionModel):
        unet = UNetMotionModel.from_unet2d(unet, motion_adapter)
    self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=
        tokenizer, unet=unet, motion_adapter=motion_adapter, scheduler=
        scheduler, feature_extractor=feature_extractor, image_encoder=
        image_encoder)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.video_processor = VideoProcessor(do_resize=False, vae_scale_factor
        =self.vae_scale_factor)
