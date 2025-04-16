def __init__(self, unet, scheduler):
    super().__init__()
    scheduler = DDIMScheduler.from_config(scheduler.config)
    self.register_modules(unet=unet, scheduler=scheduler)
