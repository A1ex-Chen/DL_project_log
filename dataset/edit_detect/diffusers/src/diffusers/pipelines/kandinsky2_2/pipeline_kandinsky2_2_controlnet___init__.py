def __init__(self, unet: UNet2DConditionModel, scheduler: DDPMScheduler,
    movq: VQModel):
    super().__init__()
    self.register_modules(unet=unet, scheduler=scheduler, movq=movq)
    self.movq_scale_factor = 2 ** (len(self.movq.config.block_out_channels) - 1
        )
