def load(self):
    model_name = self.name
    device = self.device
    model, preprocess = clip.load(model_name, device=device)
    self.preprocess = preprocess
    return model
