def __init__(self, encoder, decoder):
    super(AutoModel, self).__init__()
    self.encoder = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=500)
    self.attention = nn.Linear(500, 500)
    self.reduce = nn.Linear(500, 292)
    self.decoder = decoder
