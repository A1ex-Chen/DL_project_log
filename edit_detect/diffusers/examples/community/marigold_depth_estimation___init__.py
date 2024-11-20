def __init__(self, unet: UNet2DConditionModel, vae: AutoencoderKL,
    scheduler: DDIMScheduler, text_encoder: CLIPTextModel, tokenizer:
    CLIPTokenizer):
    super().__init__()
    self.register_modules(unet=unet, vae=vae, scheduler=scheduler,
        text_encoder=text_encoder, tokenizer=tokenizer)
    self.empty_text_embed = None
