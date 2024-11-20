def __init__(self, vgg_name, num_classes=100):
    vgg_name = vgg_name.lower()
    if vgg_name not in VGG_CFG:
        raise ValueError("Invalid identifier for vgg: '%s'" % vgg_name)
    super(VGG, self).__init__()
    self.features = nn.Sequential(*self._make_layers(VGG_CFG[vgg_name]))
    self.classifier = nn.Linear(512, num_classes)
