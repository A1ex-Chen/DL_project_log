def get_input_embeddings(self):
    """ Get model's input embeddings
        """
    base_model = getattr(self, self.base_model_prefix, self)
    if base_model is not self:
        return base_model.get_input_embeddings()
    else:
        raise NotImplementedError
