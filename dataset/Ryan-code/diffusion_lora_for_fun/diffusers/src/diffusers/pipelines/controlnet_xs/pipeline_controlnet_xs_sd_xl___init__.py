def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection, tokenizer: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer, unet: Union[UNet2DConditionModel,
    UNetControlNetXSModel], controlnet: ControlNetXSAdapter, scheduler:
    KarrasDiffusionSchedulers, force_zeros_for_empty_prompt: bool=True,
    add_watermarker: Optional[bool]=None, feature_extractor:
    CLIPImageProcessor=None):
    super().__init__()
    if isinstance(unet, UNet2DConditionModel):
        unet = UNetControlNetXSModel.from_unet(unet, controlnet)
    self.register_modules(vae=vae, text_encoder=text_encoder,
        text_encoder_2=text_encoder_2, tokenizer=tokenizer, tokenizer_2=
        tokenizer_2, unet=unet, controlnet=controlnet, scheduler=scheduler,
        feature_extractor=feature_extractor)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor, do_convert_rgb=True)
    self.control_image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor, do_convert_rgb=True, do_normalize=False)
    add_watermarker = (add_watermarker if add_watermarker is not None else
        is_invisible_watermark_available())
    if add_watermarker:
        self.watermark = StableDiffusionXLWatermarker()
    else:
        self.watermark = None
    self.register_to_config(force_zeros_for_empty_prompt=
        force_zeros_for_empty_prompt)
