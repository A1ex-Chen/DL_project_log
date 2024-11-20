def __init__(self, unet: UNet2DConditionModel, scheduler: DDPMScheduler,
    movq: VQModel, prior_prior: PriorTransformer, prior_image_encoder:
    CLIPVisionModelWithProjection, prior_text_encoder:
    CLIPTextModelWithProjection, prior_tokenizer: CLIPTokenizer,
    prior_scheduler: UnCLIPScheduler, prior_image_processor: CLIPImageProcessor
    ):
    super().__init__()
    self.register_modules(unet=unet, scheduler=scheduler, movq=movq,
        prior_prior=prior_prior, prior_image_encoder=prior_image_encoder,
        prior_text_encoder=prior_text_encoder, prior_tokenizer=
        prior_tokenizer, prior_scheduler=prior_scheduler,
        prior_image_processor=prior_image_processor)
    self.prior_pipe = KandinskyV22PriorPipeline(prior=prior_prior,
        image_encoder=prior_image_encoder, text_encoder=prior_text_encoder,
        tokenizer=prior_tokenizer, scheduler=prior_scheduler,
        image_processor=prior_image_processor)
    self.decoder_pipe = KandinskyV22InpaintPipeline(unet=unet, scheduler=
        scheduler, movq=movq)
