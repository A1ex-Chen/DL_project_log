def __init__(self, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel,
    decoder: WuerstchenDiffNeXt, scheduler: DDPMWuerstchenScheduler, vqgan:
    PaellaVQModel, latent_dim_scale: float=10.67) ->None:
    super().__init__()
    self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder,
        decoder=decoder, scheduler=scheduler, vqgan=vqgan)
    self.register_to_config(latent_dim_scale=latent_dim_scale)
