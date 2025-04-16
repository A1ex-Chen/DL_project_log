def __init__(self, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel,
    decoder: WuerstchenDiffNeXt, scheduler: DDPMWuerstchenScheduler, vqgan:
    PaellaVQModel, prior_tokenizer: CLIPTokenizer, prior_text_encoder:
    CLIPTextModel, prior_prior: WuerstchenPrior, prior_scheduler:
    DDPMWuerstchenScheduler):
    super().__init__()
    self.register_modules(text_encoder=text_encoder, tokenizer=tokenizer,
        decoder=decoder, scheduler=scheduler, vqgan=vqgan, prior_prior=
        prior_prior, prior_text_encoder=prior_text_encoder, prior_tokenizer
        =prior_tokenizer, prior_scheduler=prior_scheduler)
    self.prior_pipe = WuerstchenPriorPipeline(prior=prior_prior,
        text_encoder=prior_text_encoder, tokenizer=prior_tokenizer,
        scheduler=prior_scheduler)
    self.decoder_pipe = WuerstchenDecoderPipeline(text_encoder=text_encoder,
        tokenizer=tokenizer, decoder=decoder, scheduler=scheduler, vqgan=vqgan)
