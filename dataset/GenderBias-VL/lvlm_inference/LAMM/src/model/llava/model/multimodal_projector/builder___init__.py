def __init__(self, channels):
    super().__init__()
    self.pre_norm = nn.LayerNorm(channels)
    self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.
        Linear(channels, channels))
