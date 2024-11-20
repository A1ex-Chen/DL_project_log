@property
def dummy_feature(self):
    return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.
        dtype)
