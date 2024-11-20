def __init__(self, vae: AutoencoderKL, text_encoder: Union[
    ClapTextModelWithProjection, ClapModel], tokenizer: Union[
    RobertaTokenizer, RobertaTokenizerFast], feature_extractor: Optional[
    ClapFeatureExtractor], unet: UNet2DConditionModel, scheduler:
    KarrasDiffusionSchedulers, vocoder: SpeechT5HifiGan):
    super().__init__()
    self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=
        tokenizer, feature_extractor=feature_extractor, unet=unet,
        scheduler=scheduler, vocoder=vocoder)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
