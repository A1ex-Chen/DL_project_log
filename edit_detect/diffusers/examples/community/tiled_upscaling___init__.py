def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, low_res_scheduler:
    DDPMScheduler, scheduler: Union[DDIMScheduler, PNDMScheduler,
    LMSDiscreteScheduler], max_noise_level: int=350):
    super().__init__(vae=vae, text_encoder=text_encoder, tokenizer=
        tokenizer, unet=unet, low_res_scheduler=low_res_scheduler,
        scheduler=scheduler, max_noise_level=max_noise_level)
