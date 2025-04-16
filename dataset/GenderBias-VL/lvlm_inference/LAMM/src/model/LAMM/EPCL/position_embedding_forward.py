def forward(self, xyz, num_channels=None, input_range=None):
    assert isinstance(xyz, torch.Tensor)
    assert xyz.ndim == 3
    if self.pos_type == 'sine':
        with torch.no_grad():
            return self.get_sine_embeddings(xyz, num_channels, input_range)
    elif self.pos_type == 'fourier':
        with torch.no_grad():
            return self.get_fourier_embeddings(xyz, num_channels, input_range)
    else:
        raise ValueError(f'Unknown {self.pos_type}')
