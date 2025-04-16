def set_model_attributes(self):
    """To set or update model parameters before training."""
    self.model.names = self.data['names']
