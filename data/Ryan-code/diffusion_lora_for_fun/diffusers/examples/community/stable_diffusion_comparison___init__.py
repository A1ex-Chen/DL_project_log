def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, scheduler: Union[
    DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler], safety_checker:
    StableDiffusionSafetyChecker, feature_extractor: CLIPImageProcessor,
    requires_safety_checker: bool=True):
    super()._init_()
    self.pipe1 = StableDiffusionPipeline.from_pretrained(pipe1_model_id)
    self.pipe2 = StableDiffusionPipeline.from_pretrained(pipe2_model_id)
    self.pipe3 = StableDiffusionPipeline.from_pretrained(pipe3_model_id)
    self.pipe4 = StableDiffusionPipeline(vae=vae, text_encoder=text_encoder,
        tokenizer=tokenizer, unet=unet, scheduler=scheduler, safety_checker
        =safety_checker, feature_extractor=feature_extractor,
        requires_safety_checker=requires_safety_checker)
    self.register_modules(pipeline1=self.pipe1, pipeline2=self.pipe2,
        pipeline3=self.pipe3, pipeline4=self.pipe4)
