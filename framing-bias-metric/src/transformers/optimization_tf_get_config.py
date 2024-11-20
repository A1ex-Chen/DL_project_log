def get_config(self):
    config = super().get_config()
    config.update({'weight_decay_rate': self.weight_decay_rate})
    return config
