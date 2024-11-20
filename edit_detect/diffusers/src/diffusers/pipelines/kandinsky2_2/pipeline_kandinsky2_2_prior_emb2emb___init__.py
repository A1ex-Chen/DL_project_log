def __init__(self, prior: PriorTransformer, image_encoder:
    CLIPVisionModelWithProjection, text_encoder:
    CLIPTextModelWithProjection, tokenizer: CLIPTokenizer, scheduler:
    UnCLIPScheduler, image_processor: CLIPImageProcessor):
    super().__init__()
    self.register_modules(prior=prior, text_encoder=text_encoder, tokenizer
        =tokenizer, scheduler=scheduler, image_encoder=image_encoder,
        image_processor=image_processor)
