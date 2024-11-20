def __init__(self, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel,
    decoder: StableCascadeUNet, scheduler: DDPMWuerstchenScheduler, vqgan:
    PaellaVQModel, prior_prior: StableCascadeUNet, prior_text_encoder:
    CLIPTextModel, prior_tokenizer: CLIPTokenizer, prior_scheduler:
    DDPMWuerstchenScheduler, prior_feature_extractor: Optional[
    CLIPImageProcessor]=None, prior_image_encoder: Optional[
    CLIPVisionModelWithProjection]=None):
    super().__init__()
    self.register_modules(text_encoder=text_encoder, tokenizer=tokenizer,
        decoder=decoder, scheduler=scheduler, vqgan=vqgan,
        prior_text_encoder=prior_text_encoder, prior_tokenizer=
        prior_tokenizer, prior_prior=prior_prior, prior_scheduler=
        prior_scheduler, prior_feature_extractor=prior_feature_extractor,
        prior_image_encoder=prior_image_encoder)
    self.prior_pipe = StableCascadePriorPipeline(prior=prior_prior,
        text_encoder=prior_text_encoder, tokenizer=prior_tokenizer,
        scheduler=prior_scheduler, image_encoder=prior_image_encoder,
        feature_extractor=prior_feature_extractor)
    self.decoder_pipe = StableCascadeDecoderPipeline(text_encoder=
        text_encoder, tokenizer=tokenizer, decoder=decoder, scheduler=
        scheduler, vqgan=vqgan)
