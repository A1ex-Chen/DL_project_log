def __init__(self, unet: UNet2DModel, scheduler: ScoreSdeVeScheduler):
    super().__init__()
    self.register_modules(unet=unet, scheduler=scheduler)
