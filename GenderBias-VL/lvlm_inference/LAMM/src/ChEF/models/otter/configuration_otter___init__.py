def __init__(self, vision_config=None, text_config=None,
    cross_attn_every_n_layers: int=4, use_media_placement_augmentation:
    bool=True, only_attend_previous: bool=True, **kwargs):
    super().__init__(**kwargs)
    if vision_config is None:
        vision_config = {}
        logger.info(
            'vision_config is None. initializing the vision config with default values.'
            )
    if text_config is None:
        text_config = {}
        logger.info(
            'text_config is None. Initializing the text config with default values.'
            )
    self.vision_config = CLIPVisionConfig(**vision_config)
    self.text_config = CONFIG_MAPPING[text_config.pop('model_type')](**
        text_config)
    self.cross_attn_every_n_layers = cross_attn_every_n_layers
    self.use_media_placement_augmentation = use_media_placement_augmentation
    self.only_attend_previous = only_attend_previous
