def __init__(self, tokenizer: CLIPTokenizer, text_encoder:
    ContextCLIPTextModel, vae: AutoencoderKL, unet: UNet2DConditionModel,
    scheduler: PNDMScheduler, qformer: Blip2QFormerModel, image_processor:
    BlipImageProcessor, ctx_begin_pos: int=2, mean: List[float]=None, std:
    List[float]=None):
    super().__init__()
    self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder,
        vae=vae, unet=unet, scheduler=scheduler, qformer=qformer,
        image_processor=image_processor)
    self.register_to_config(ctx_begin_pos=ctx_begin_pos, mean=mean, std=std)
