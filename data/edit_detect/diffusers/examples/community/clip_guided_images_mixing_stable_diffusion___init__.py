def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    clip_model: CLIPModel, tokenizer: CLIPTokenizer, unet:
    UNet2DConditionModel, scheduler: Union[PNDMScheduler,
    LMSDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler],
    feature_extractor: CLIPFeatureExtractor, coca_model=None,
    coca_tokenizer=None, coca_transform=None):
    super().__init__()
    self.register_modules(vae=vae, text_encoder=text_encoder, clip_model=
        clip_model, tokenizer=tokenizer, unet=unet, scheduler=scheduler,
        feature_extractor=feature_extractor, coca_model=coca_model,
        coca_tokenizer=coca_tokenizer, coca_transform=coca_transform)
    self.feature_extractor_size = feature_extractor.size if isinstance(
        feature_extractor.size, int) else feature_extractor.size[
        'shortest_edge']
    self.normalize = transforms.Normalize(mean=feature_extractor.image_mean,
        std=feature_extractor.image_std)
    set_requires_grad(self.text_encoder, False)
    set_requires_grad(self.clip_model, False)
