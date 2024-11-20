def __init__(self, text_encoder: FlaxCLIPTextModel, text_encoder_2:
    FlaxCLIPTextModel, vae: FlaxAutoencoderKL, tokenizer: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer, unet: FlaxUNet2DConditionModel, scheduler:
    Union[FlaxDDIMScheduler, FlaxPNDMScheduler, FlaxLMSDiscreteScheduler,
    FlaxDPMSolverMultistepScheduler], dtype: jnp.dtype=jnp.float32):
    super().__init__()
    self.dtype = dtype
    self.register_modules(vae=vae, text_encoder=text_encoder,
        text_encoder_2=text_encoder_2, tokenizer=tokenizer, tokenizer_2=
        tokenizer_2, unet=unet, scheduler=scheduler)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
