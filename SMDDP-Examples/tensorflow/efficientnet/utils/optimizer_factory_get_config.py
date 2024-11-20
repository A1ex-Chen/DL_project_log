def get_config(self):
    config = {'optimizer': tf.keras.optimizers.serialize(self._optimizer),
        'average_decay': self._average_decay, 'start_step': self.
        _start_step, 'dynamic_decay': self._dynamic_decay}
    base_config = super(MovingAverage, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
