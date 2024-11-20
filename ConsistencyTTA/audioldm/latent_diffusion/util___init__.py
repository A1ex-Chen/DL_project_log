def __init__(self, c_concat_config, c_crossattn_config):
    super().__init__()
    self.concat_conditioner = instantiate_from_config(c_concat_config)
    self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)
