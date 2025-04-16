def __init__(self, prior: PriorTransformer, text_encoder:
    CLIPTextModelWithProjection, tokenizer: CLIPTokenizer, scheduler:
    HeunDiscreteScheduler, shap_e_renderer: ShapERenderer):
    super().__init__()
    self.register_modules(prior=prior, text_encoder=text_encoder, tokenizer
        =tokenizer, scheduler=scheduler, shap_e_renderer=shap_e_renderer)
