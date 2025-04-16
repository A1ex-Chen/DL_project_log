def get_config(self):
    config = {'use_scale': self.use_scale}
    base_config = super(AdditiveAttention, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
