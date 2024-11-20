def __init__(self, visual_config=None, **kwargs):
    if visual_config is None:
        self.visual_config = DEFAULT_VISUAL_CONFIG
    else:
        self.visual_config = visual_config
    super().__init__(**kwargs)
