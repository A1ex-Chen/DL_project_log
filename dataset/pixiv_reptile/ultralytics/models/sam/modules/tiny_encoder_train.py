@torch.no_grad()
def train(self, mode=True):
    """Sets the module in training mode and handles attribute 'ab' based on the mode."""
    super().train(mode)
    if mode and hasattr(self, 'ab'):
        del self.ab
    else:
        self.ab = self.attention_biases[:, self.attention_bias_idxs]
