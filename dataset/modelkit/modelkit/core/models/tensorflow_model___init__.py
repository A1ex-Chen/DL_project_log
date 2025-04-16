def __init__(self, saved_model_signature):
    super().__init__()
    self.layer = LayerFromSavedModel(saved_model_signature)
