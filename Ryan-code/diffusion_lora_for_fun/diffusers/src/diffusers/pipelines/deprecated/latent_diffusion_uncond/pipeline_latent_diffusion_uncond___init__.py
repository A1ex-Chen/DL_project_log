def __init__(self, vqvae: VQModel, unet: UNet2DModel, scheduler: DDIMScheduler
    ):
    super().__init__()
    self.register_modules(vqvae=vqvae, unet=unet, scheduler=scheduler)
