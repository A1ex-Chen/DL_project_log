def __init__(self):
    super().__init__()
    if args.encoder == 'resnet34':
        Encoder = ResNet34
    elif args.encoder == 'resnet18':
        Encoder = ResNet18
    self.encoder = Encoder(block='NonBottleneck1D', pretrained_on_imagenet=
        False, activation=nn.ReLU(inplace=True))
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, 1000)
