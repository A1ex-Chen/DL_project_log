def load_model(self, model_path):
    """Loads a trained model from a .pth file."""
    self.model.load_state_dict(torch.load(model_path, weights_only=True))
    self.model.eval()
