def __init__(self, unet: UNet2DModel, scheduler: CMStochasticIterativeScheduler
    ) ->None:
    super().__init__()
    self.register_modules(unet=unet, scheduler=scheduler)
    self.safety_checker = None
