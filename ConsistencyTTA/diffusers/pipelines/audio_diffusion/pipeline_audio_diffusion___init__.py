def __init__(self, vqvae: AutoencoderKL, unet: UNet2DConditionModel, mel:
    Mel, scheduler: Union[DDIMScheduler, DDPMScheduler]):
    super().__init__()
    self.register_modules(unet=unet, scheduler=scheduler, mel=mel, vqvae=vqvae)
