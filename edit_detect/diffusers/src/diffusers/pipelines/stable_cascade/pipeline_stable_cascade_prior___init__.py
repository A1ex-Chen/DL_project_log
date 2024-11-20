def __init__(self, tokenizer: CLIPTokenizer, text_encoder:
    CLIPTextModelWithProjection, prior: StableCascadeUNet, scheduler:
    DDPMWuerstchenScheduler, resolution_multiple: float=42.67,
    feature_extractor: Optional[CLIPImageProcessor]=None, image_encoder:
    Optional[CLIPVisionModelWithProjection]=None) ->None:
    super().__init__()
    self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder,
        image_encoder=image_encoder, feature_extractor=feature_extractor,
        prior=prior, scheduler=scheduler)
    self.register_to_config(resolution_multiple=resolution_multiple)
