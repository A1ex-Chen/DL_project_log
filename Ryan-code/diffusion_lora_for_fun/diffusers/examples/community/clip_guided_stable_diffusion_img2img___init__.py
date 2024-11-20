def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    clip_model: CLIPModel, tokenizer: CLIPTokenizer, unet:
    UNet2DConditionModel, scheduler: Union[PNDMScheduler,
    LMSDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler],
    feature_extractor: CLIPFeatureExtractor):
    super().__init__()
    self.register_modules(vae=vae, text_encoder=text_encoder, clip_model=
        clip_model, tokenizer=tokenizer, unet=unet, scheduler=scheduler,
        feature_extractor=feature_extractor)
    self.normalize = transforms.Normalize(mean=feature_extractor.image_mean,
        std=feature_extractor.image_std)
    self.cut_out_size = feature_extractor.size if isinstance(feature_extractor
        .size, int) else feature_extractor.size['shortest_edge']
    self.make_cutouts = MakeCutouts(self.cut_out_size)
    set_requires_grad(self.text_encoder, False)
    set_requires_grad(self.clip_model, False)
