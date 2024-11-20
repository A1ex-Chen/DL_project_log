def load_weights(model: tf.keras.Model, model_weights_path: Text,
    weights_format: Text='saved_model'):
    """Load model weights from the given file path.

  Args:
    model: the model to load weights into
    model_weights_path: the path of the model weights
    weights_format: the model weights format. One of 'saved_model', 'h5',
       or 'checkpoint'.
  """
    if weights_format == 'saved_model':
        loaded_model = tf.keras.models.load_model(model_weights_path)
        model.set_weights(loaded_model.get_weights())
    else:
        model.load_weights(model_weights_path)
