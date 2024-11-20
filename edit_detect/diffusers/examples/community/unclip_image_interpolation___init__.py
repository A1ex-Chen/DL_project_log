def __init__(self, decoder: UNet2DConditionModel, text_encoder:
    CLIPTextModelWithProjection, tokenizer: CLIPTokenizer, text_proj:
    UnCLIPTextProjModel, feature_extractor: CLIPImageProcessor,
    image_encoder: CLIPVisionModelWithProjection, super_res_first:
    UNet2DModel, super_res_last: UNet2DModel, decoder_scheduler:
    UnCLIPScheduler, super_res_scheduler: UnCLIPScheduler):
    super().__init__()
    self.register_modules(decoder=decoder, text_encoder=text_encoder,
        tokenizer=tokenizer, text_proj=text_proj, feature_extractor=
        feature_extractor, image_encoder=image_encoder, super_res_first=
        super_res_first, super_res_last=super_res_last, decoder_scheduler=
        decoder_scheduler, super_res_scheduler=super_res_scheduler)
