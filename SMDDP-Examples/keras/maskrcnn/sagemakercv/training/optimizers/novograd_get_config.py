def get_config(self):
    config = super().get_config()
    config.update({'learning_rate': self._serialize_hyperparameter(
        'learning_rate'), 'beta_1': self._serialize_hyperparameter('beta_1'
        ), 'beta_2': self._serialize_hyperparameter('beta_2'), 'epsilon':
        self.epsilon, 'weight_decay': self._serialize_hyperparameter(
        'weight_decay'), 'grad_averaging': self._serialize_hyperparameter(
        'grad_averaging')})
    return config
