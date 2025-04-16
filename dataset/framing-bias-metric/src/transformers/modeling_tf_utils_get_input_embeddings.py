def get_input_embeddings(self) ->tf.keras.layers.Layer:
    """
        Returns the model's input embeddings.

        Returns:
            :obj:`tf.keras.layers.Layer`: A torch module mapping vocabulary to hidden states.
        """
    base_model = getattr(self, self.base_model_prefix, self)
    if base_model is not self:
        return base_model.get_input_embeddings()
    else:
        raise NotImplementedError
