def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    image_encoder: CLIPVisionModelWithProjection, clip_image_processor:
    CLIPImageProcessor, clip_tokenizer: CLIPTokenizer, text_decoder:
    UniDiffuserTextDecoder, text_tokenizer: GPT2Tokenizer, unet:
    UniDiffuserModel, scheduler: KarrasDiffusionSchedulers):
    super().__init__()
    if text_encoder.config.hidden_size != text_decoder.prefix_inner_dim:
        raise ValueError(
            f'The text encoder hidden size and text decoder prefix inner dim must be the same, but `text_encoder.config.hidden_size`: {text_encoder.config.hidden_size} and `text_decoder.prefix_inner_dim`: {text_decoder.prefix_inner_dim}'
            )
    self.register_modules(vae=vae, text_encoder=text_encoder, image_encoder
        =image_encoder, clip_image_processor=clip_image_processor,
        clip_tokenizer=clip_tokenizer, text_decoder=text_decoder,
        text_tokenizer=text_tokenizer, unet=unet, scheduler=scheduler)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor)
    self.num_channels_latents = vae.config.latent_channels
    self.text_encoder_seq_len = text_encoder.config.max_position_embeddings
    self.text_encoder_hidden_size = text_encoder.config.hidden_size
    self.image_encoder_projection_dim = image_encoder.config.projection_dim
    self.unet_resolution = unet.config.sample_size
    self.text_intermediate_dim = self.text_encoder_hidden_size
    if self.text_decoder.prefix_hidden_dim is not None:
        self.text_intermediate_dim = self.text_decoder.prefix_hidden_dim
    self.mode = None
    self.safety_checker = None
