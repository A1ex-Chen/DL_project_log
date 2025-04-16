def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, motion_adapter:
    MotionAdapter, controlnet: Union[ControlNetModel, List[ControlNetModel],
    Tuple[ControlNetModel], MultiControlNetModel], scheduler: Union[
    DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler,
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler], feature_extractor: Optional[
    CLIPImageProcessor]=None, image_encoder: Optional[
    CLIPVisionModelWithProjection]=None):
    super().__init__()
    unet = UNetMotionModel.from_unet2d(unet, motion_adapter)
    if isinstance(controlnet, (list, tuple)):
        controlnet = MultiControlNetModel(controlnet)
    self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=
        tokenizer, unet=unet, motion_adapter=motion_adapter, controlnet=
        controlnet, scheduler=scheduler, feature_extractor=
        feature_extractor, image_encoder=image_encoder)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor)
    self.control_image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor, do_convert_rgb=True, do_normalize=False)
