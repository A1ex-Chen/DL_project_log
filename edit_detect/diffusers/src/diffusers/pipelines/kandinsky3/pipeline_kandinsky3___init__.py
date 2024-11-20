def __init__(self, tokenizer: T5Tokenizer, text_encoder: T5EncoderModel,
    unet: Kandinsky3UNet, scheduler: DDPMScheduler, movq: VQModel):
    super().__init__()
    self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder,
        unet=unet, scheduler=scheduler, movq=movq)
