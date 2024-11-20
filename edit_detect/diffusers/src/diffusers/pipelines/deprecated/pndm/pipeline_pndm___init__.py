def __init__(self, unet: UNet2DModel, scheduler: PNDMScheduler):
    super().__init__()
    scheduler = PNDMScheduler.from_config(scheduler.config)
    self.register_modules(unet=unet, scheduler=scheduler)
