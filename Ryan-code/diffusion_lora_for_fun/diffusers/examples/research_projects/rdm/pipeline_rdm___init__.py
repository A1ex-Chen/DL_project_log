def __init__(self, vae: AutoencoderKL, clip: CLIPModel, tokenizer:
    CLIPTokenizer, unet: UNet2DConditionModel, scheduler: Union[
    DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler,
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler], feature_extractor: CLIPFeatureExtractor,
    retriever: Optional[Retriever]=None):
    super().__init__()
    self.register_modules(vae=vae, clip=clip, tokenizer=tokenizer, unet=
        unet, scheduler=scheduler, feature_extractor=feature_extractor)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor)
    self.retriever = retriever
