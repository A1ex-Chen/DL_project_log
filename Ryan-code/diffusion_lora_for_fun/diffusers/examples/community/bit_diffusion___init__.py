def __init__(self, unet: UNet2DConditionModel, scheduler: Union[
    DDIMScheduler, DDPMScheduler], bit_scale: Optional[float]=1.0):
    super().__init__()
    self.bit_scale = bit_scale
    self.scheduler.step = ddim_bit_scheduler_step if isinstance(scheduler,
        DDIMScheduler) else ddpm_bit_scheduler_step
    self.register_modules(unet=unet, scheduler=scheduler)
