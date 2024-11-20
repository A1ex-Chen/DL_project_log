def __init__(self, config):
    super(LlavaMetaModel, self).__init__(config)
    if hasattr(config, 'mm_vision_tower'):
        self.vision_tower = build_vision_tower(config, delay_load=True)
        self.mm_projector = build_vision_projector(config)
