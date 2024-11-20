def __init__(self, pad_mode: str='reflect'):
    super().__init__()
    self.pad_mode = pad_mode
    kernel_1d = torch.tensor([[1 / 8, 3 / 8, 3 / 8, 1 / 8]])
    self.pad = kernel_1d.shape[1] // 2 - 1
    self.register_buffer('kernel', kernel_1d.T @ kernel_1d, persistent=False)
