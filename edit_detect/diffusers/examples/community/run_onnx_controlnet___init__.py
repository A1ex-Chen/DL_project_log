def __init__(self, vae_encoder: OnnxRuntimeModel, vae_decoder:
    OnnxRuntimeModel, text_encoder: OnnxRuntimeModel, tokenizer:
    CLIPTokenizer, unet: OnnxRuntimeModel, scheduler: KarrasDiffusionSchedulers
    ):
    super().__init__()
    self.register_modules(vae_encoder=vae_encoder, vae_decoder=vae_decoder,
        text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=scheduler)
    self.vae_scale_factor = 2 ** (4 - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor, do_convert_rgb=True)
    self.control_image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor, do_convert_rgb=True, do_normalize=False)
