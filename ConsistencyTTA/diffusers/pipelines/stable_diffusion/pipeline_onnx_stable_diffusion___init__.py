def __init__(self, vae_encoder: OnnxRuntimeModel, vae_decoder:
    OnnxRuntimeModel, text_encoder: OnnxRuntimeModel, tokenizer:
    CLIPTokenizer, unet: OnnxRuntimeModel, scheduler: Union[DDIMScheduler,
    PNDMScheduler, LMSDiscreteScheduler], safety_checker: OnnxRuntimeModel,
    feature_extractor: CLIPImageProcessor):
    deprecation_message = (
        'Please use `OnnxStableDiffusionPipeline` instead of `StableDiffusionOnnxPipeline`.'
        )
    deprecate('StableDiffusionOnnxPipeline', '1.0.0', deprecation_message)
    super().__init__(vae_encoder=vae_encoder, vae_decoder=vae_decoder,
        text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=scheduler, safety_checker=safety_checker,
        feature_extractor=feature_extractor)
