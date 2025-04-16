def __init__(self, unet, scheduler, inverse_scheduler=None):
    super().__init__()
    self.register_modules(unet=unet, scheduler=scheduler)
    self.inverse_scheduler = inverse_scheduler
