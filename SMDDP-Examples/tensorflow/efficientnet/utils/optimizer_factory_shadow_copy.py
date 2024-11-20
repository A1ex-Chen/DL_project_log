def shadow_copy(self, model: tf.keras.Model):
    """Creates shadow variables for the given model weights."""
    for var in model.weights:
        self.add_slot(var, 'average', initializer='zeros')
    self._average_weights = [self.get_slot(var, 'average') for var in model
        .weights]
    self._model_weights = model.weights
