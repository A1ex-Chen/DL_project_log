def __init__(self, prior: PriorTransformer, tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModelWithProjection, prior_scheduler:
    UnCLIPScheduler, decoder_pipe_kwargs: Optional[dict]=None):
    super().__init__()
    decoder_pipe_kwargs = {'image_encoder': None
        } if decoder_pipe_kwargs is None else decoder_pipe_kwargs
    decoder_pipe_kwargs['torch_dtype'] = decoder_pipe_kwargs.get('torch_dtype',
        None) or prior.dtype
    self.decoder_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        'lambdalabs/sd-image-variations-diffusers', **decoder_pipe_kwargs)
    self.decoder_pipe._encode_image = types.MethodType(_encode_image, self.
        decoder_pipe)
    self.register_modules(prior=prior, tokenizer=tokenizer, text_encoder=
        text_encoder, prior_scheduler=prior_scheduler)
