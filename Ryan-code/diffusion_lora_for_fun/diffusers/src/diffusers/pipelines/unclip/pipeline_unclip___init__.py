def __init__(self, prior: PriorTransformer, decoder: UNet2DConditionModel,
    text_encoder: CLIPTextModelWithProjection, tokenizer: CLIPTokenizer,
    text_proj: UnCLIPTextProjModel, super_res_first: UNet2DModel,
    super_res_last: UNet2DModel, prior_scheduler: UnCLIPScheduler,
    decoder_scheduler: UnCLIPScheduler, super_res_scheduler: UnCLIPScheduler):
    super().__init__()
    self.register_modules(prior=prior, decoder=decoder, text_encoder=
        text_encoder, tokenizer=tokenizer, text_proj=text_proj,
        super_res_first=super_res_first, super_res_last=super_res_last,
        prior_scheduler=prior_scheduler, decoder_scheduler=
        decoder_scheduler, super_res_scheduler=super_res_scheduler)
