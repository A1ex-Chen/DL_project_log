def set_model(self, model):
    if isinstance(model.layers[-2], Model):
        self.model = model.layers[-2]
    else:
        self.model = model
