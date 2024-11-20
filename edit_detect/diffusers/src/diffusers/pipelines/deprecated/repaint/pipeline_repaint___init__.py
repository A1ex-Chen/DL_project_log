def __init__(self, unet, scheduler):
    super().__init__()
    self.register_modules(unet=unet, scheduler=scheduler)
