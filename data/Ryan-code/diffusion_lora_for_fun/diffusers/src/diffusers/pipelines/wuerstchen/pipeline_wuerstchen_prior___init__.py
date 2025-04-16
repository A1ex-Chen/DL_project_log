def __init__(self, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel,
    prior: WuerstchenPrior, scheduler: DDPMWuerstchenScheduler, latent_mean:
    float=42.0, latent_std: float=1.0, resolution_multiple: float=42.67
    ) ->None:
    super().__init__()
    self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder,
        prior=prior, scheduler=scheduler)
    self.register_to_config(latent_mean=latent_mean, latent_std=latent_std,
        resolution_multiple=resolution_multiple)
