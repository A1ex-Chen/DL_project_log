def get_config(self):
    config = super(MomentumOptimizer, self).get_config()
    config.update({'learning_rate': self._serialize_hyperparameter(
        'learning_rate'), 'decay': self._serialize_hyperparameter('decay'),
        'momentum': self._serialize_hyperparameter('momentum'), 'nesterov':
        self.nesterov})
    return config
