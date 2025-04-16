def __init__(self, decoder: StableCascadeUNet, tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel, scheduler: DDPMWuerstchenScheduler, vqgan:
    PaellaVQModel, latent_dim_scale: float=10.67) ->None:
    super().__init__()
    self.register_modules(decoder=decoder, tokenizer=tokenizer,
        text_encoder=text_encoder, scheduler=scheduler, vqgan=vqgan)
    self.register_to_config(latent_dim_scale=latent_dim_scale)
