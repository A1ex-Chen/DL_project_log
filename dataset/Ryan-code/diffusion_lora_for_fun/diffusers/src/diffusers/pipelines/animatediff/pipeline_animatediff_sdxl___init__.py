def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection, tokenizer: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer, unet: Union[UNet2DConditionModel,
    UNetMotionModel], motion_adapter: MotionAdapter, scheduler: Union[
    DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler,
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler], image_encoder:
    CLIPVisionModelWithProjection=None, feature_extractor:
    CLIPImageProcessor=None, force_zeros_for_empty_prompt: bool=True):
    super().__init__()
    if isinstance(unet, UNet2DConditionModel):
        unet = UNetMotionModel.from_unet2d(unet, motion_adapter)
    self.register_modules(vae=vae, text_encoder=text_encoder,
        text_encoder_2=text_encoder_2, tokenizer=tokenizer, tokenizer_2=
        tokenizer_2, unet=unet, motion_adapter=motion_adapter, scheduler=
        scheduler, image_encoder=image_encoder, feature_extractor=
        feature_extractor)
    self.register_to_config(force_zeros_for_empty_prompt=
        force_zeros_for_empty_prompt)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.video_processor = VideoProcessor(vae_scale_factor=self.
        vae_scale_factor)
    self.default_sample_size = self.unet.config.sample_size
