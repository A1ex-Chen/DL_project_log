def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection, tokenizer: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer, unet: UNet2DConditionModel, adapter: Union[
    T2IAdapter, MultiAdapter, List[T2IAdapter]], controlnet: Union[
    ControlNetModel, MultiControlNetModel], scheduler:
    KarrasDiffusionSchedulers, force_zeros_for_empty_prompt: bool=True):
    super().__init__()
    if isinstance(controlnet, (list, tuple)):
        controlnet = MultiControlNetModel(controlnet)
    self.register_modules(vae=vae, text_encoder=text_encoder,
        text_encoder_2=text_encoder_2, tokenizer=tokenizer, tokenizer_2=
        tokenizer_2, unet=unet, adapter=adapter, controlnet=controlnet,
        scheduler=scheduler)
    self.register_to_config(force_zeros_for_empty_prompt=
        force_zeros_for_empty_prompt)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor)
    self.control_image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor, do_convert_rgb=True, do_normalize=False)
    self.default_sample_size = self.unet.config.sample_size
