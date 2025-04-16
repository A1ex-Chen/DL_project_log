@tf.function
def _predict_step(self, text):
    predictions = self.model(text, training=False)
    return predictions
