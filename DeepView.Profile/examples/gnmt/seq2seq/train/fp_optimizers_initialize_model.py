def initialize_model(self, model):
    """
        Initializes state of the model.

        :param model: model
        """
    self.model = model
    self.model.zero_grad()
