def __init__(self, unet: UNet2DModel, scheduler: KarrasVeScheduler):
    super().__init__()
    self.register_modules(unet=unet, scheduler=scheduler)
