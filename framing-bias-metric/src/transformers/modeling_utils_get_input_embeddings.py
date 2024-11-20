def get_input_embeddings(self) ->nn.Module:
    """
        Returns the model's input embeddings.

        Returns:
            :obj:`nn.Module`: A torch module mapping vocabulary to hidden states.
        """
    base_model = getattr(self, self.base_model_prefix, self)
    if base_model is not self:
        return base_model.get_input_embeddings()
    else:
        raise NotImplementedError
