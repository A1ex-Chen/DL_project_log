def __init__(self, vision_tower, args, delay_load=False):
    super().__init__()
    self.is_loaded = False
    self.vision_tower_name = vision_tower
    self.select_layer = args.mm_vision_select_layer
    self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
    if not delay_load:
        self.load_model()
    else:
        self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name
            )
