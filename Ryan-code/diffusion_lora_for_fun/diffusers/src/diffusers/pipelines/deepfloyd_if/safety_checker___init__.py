def __init__(self, config: CLIPConfig):
    super().__init__(config)
    self.vision_model = CLIPVisionModelWithProjection(config.vision_config)
    self.p_head = nn.Linear(config.vision_config.projection_dim, 1)
    self.w_head = nn.Linear(config.vision_config.projection_dim, 1)
