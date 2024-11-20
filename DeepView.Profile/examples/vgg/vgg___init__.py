def __init__(self, cfg, batch_norm=False, num_classes=1000, init_weights=True):
    super(VGG, self).__init__()
    self.features = self.make_layers(cfg, batch_norm)
    self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
    self.lin1 = nn.Linear(512 * 7 * 7, 4096)
    self.relu = nn.ReLU(True)
    self.dropout = nn.Dropout()
    self.lin2 = nn.Linear(4096, 4096)
    self.lin3 = nn.Linear(4096, num_classes)
    self.loss_fn = nn.CrossEntropyLoss()
    if init_weights:
        self._initialize_weights()
