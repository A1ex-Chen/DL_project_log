def get_config(self):
    config = super().get_config()
    config.update({'weight_decay': self._serialize_hyperparameter(
        'weight_decay')})
    return config
