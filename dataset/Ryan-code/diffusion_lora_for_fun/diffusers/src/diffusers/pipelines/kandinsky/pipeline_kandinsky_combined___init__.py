def __init__(self, text_encoder: MultilingualCLIP, tokenizer:
    XLMRobertaTokenizer, unet: UNet2DConditionModel, scheduler: Union[
    DDIMScheduler, DDPMScheduler], movq: VQModel, prior_prior:
    PriorTransformer, prior_image_encoder: CLIPVisionModelWithProjection,
    prior_text_encoder: CLIPTextModelWithProjection, prior_tokenizer:
    CLIPTokenizer, prior_scheduler: UnCLIPScheduler, prior_image_processor:
    CLIPImageProcessor):
    super().__init__()
    self.register_modules(text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, scheduler=scheduler, movq=movq, prior_prior=prior_prior,
        prior_image_encoder=prior_image_encoder, prior_text_encoder=
        prior_text_encoder, prior_tokenizer=prior_tokenizer,
        prior_scheduler=prior_scheduler, prior_image_processor=
        prior_image_processor)
    self.prior_pipe = KandinskyPriorPipeline(prior=prior_prior,
        image_encoder=prior_image_encoder, text_encoder=prior_text_encoder,
        tokenizer=prior_tokenizer, scheduler=prior_scheduler,
        image_processor=prior_image_processor)
    self.decoder_pipe = KandinskyInpaintPipeline(text_encoder=text_encoder,
        tokenizer=tokenizer, unet=unet, scheduler=scheduler, movq=movq)
