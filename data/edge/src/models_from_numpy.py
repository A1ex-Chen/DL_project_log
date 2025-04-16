def from_numpy(self, x):
    return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray
        ) else x
