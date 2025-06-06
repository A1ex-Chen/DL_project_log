def __init__(self, text_encoder: MultilingualCLIP, tokenizer:
    XLMRobertaTokenizer, unet: UNet2DConditionModel, scheduler: Union[
    DDIMScheduler, DDPMScheduler], movq: VQModel):
    super().__init__()
    self.register_modules(text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, scheduler=scheduler, movq=movq)
    self.movq_scale_factor = 2 ** (len(self.movq.config.block_out_channels) - 1
        )
