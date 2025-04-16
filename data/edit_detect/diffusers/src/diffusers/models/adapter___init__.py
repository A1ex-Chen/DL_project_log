def __init__(self, channels: int):
    super().__init__()
    self.block1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.act = nn.ReLU()
    self.block2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
