def __init__(self, prior: PriorTransformer, image_encoder: CLIPVisionModel,
    image_processor: CLIPImageProcessor, scheduler: HeunDiscreteScheduler,
    shap_e_renderer: ShapERenderer):
    super().__init__()
    self.register_modules(prior=prior, image_encoder=image_encoder,
        image_processor=image_processor, scheduler=scheduler,
        shap_e_renderer=shap_e_renderer)
