def num_params(self):
    """Get the number of model parameters."""
    return sum(p.numel() for p in self.parameters())
