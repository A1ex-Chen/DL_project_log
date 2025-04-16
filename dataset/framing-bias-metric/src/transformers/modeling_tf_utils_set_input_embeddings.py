def set_input_embeddings(self, value):
    """
        Set model's input embeddings.

        Args:
            value (:obj:`tf.keras.layers.Layer`):
                A module mapping vocabulary to hidden states.
        """
    base_model = getattr(self, self.base_model_prefix, self)
    if base_model is not self:
        base_model.set_input_embeddings(value)
    else:
        raise NotImplementedError
