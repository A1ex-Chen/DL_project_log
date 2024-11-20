def __init__(self, vae: AutoencoderKL, text_encoder: ClapModel,
    text_encoder_2: Union[T5EncoderModel, VitsModel], projection_model:
    AudioLDM2ProjectionModel, language_model: GPT2Model, tokenizer: Union[
    RobertaTokenizer, RobertaTokenizerFast], tokenizer_2: Union[T5Tokenizer,
    T5TokenizerFast, VitsTokenizer], feature_extractor:
    ClapFeatureExtractor, unet: AudioLDM2UNet2DConditionModel, scheduler:
    KarrasDiffusionSchedulers, vocoder: SpeechT5HifiGan):
    super().__init__()
    self.register_modules(vae=vae, text_encoder=text_encoder,
        text_encoder_2=text_encoder_2, projection_model=projection_model,
        language_model=language_model, tokenizer=tokenizer, tokenizer_2=
        tokenizer_2, feature_extractor=feature_extractor, unet=unet,
        scheduler=scheduler, vocoder=vocoder)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
