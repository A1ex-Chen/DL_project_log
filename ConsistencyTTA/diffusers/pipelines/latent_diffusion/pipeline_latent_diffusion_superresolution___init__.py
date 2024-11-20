def __init__(self, vqvae: VQModel, unet: UNet2DModel, scheduler: Union[
    DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler,
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler]):
    super().__init__()
    self.register_modules(vqvae=vqvae, unet=unet, scheduler=scheduler)
