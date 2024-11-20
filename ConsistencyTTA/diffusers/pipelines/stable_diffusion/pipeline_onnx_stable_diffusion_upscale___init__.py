def __init__(self, vae: OnnxRuntimeModel, text_encoder: OnnxRuntimeModel,
    tokenizer: Any, unet: OnnxRuntimeModel, low_res_scheduler:
    DDPMScheduler, scheduler: Any, max_noise_level: int=350):
    super().__init__(vae, text_encoder, tokenizer, unet, low_res_scheduler,
        scheduler, max_noise_level)
