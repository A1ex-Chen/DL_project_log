def __init__(self, config: CLIPVisionConfig):
    super().__init__(config)
    self.vision_model = CLIPVisionTransformer(config)
    self.post_init()
