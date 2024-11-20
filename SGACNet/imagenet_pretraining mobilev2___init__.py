def __init__(self):
    super().__init__()
    if args.encoder == 'mobilenet_v2':
        Encoder = mobilenet_v2
    elif args.encoder == 'mobilenet_v2':
        Encoder = mobilenet_v2
    self.encoder = Encoder(block='NonBottleneck1D', pretrained_on_imagenet=
        False, activation=nn.ReLU(inplace=True))
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, 1000)
