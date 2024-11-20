def __init__(self, model: Model) ->None:
    self.model = model
    self.main_model_name = self.model.configuration_key
    self._build(self.model)
