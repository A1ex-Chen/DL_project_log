def __init__(self):
    super(VGG11, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
        padding=1)
    self.bn1 = nn.BatchNorm2d(64)
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
        padding=1)
    self.bn2 = nn.BatchNorm2d(128)
    self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
        padding=1)
    self.bn3 = nn.BatchNorm2d(256)
    self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
        padding=1)
    self.bn4 = nn.BatchNorm2d(256)
    self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
        padding=1)
    self.bn5 = nn.BatchNorm2d(512)
    self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
        padding=1)
    self.bn6 = nn.BatchNorm2d(512)
    self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
        padding=1)
    self.bn7 = nn.BatchNorm2d(512)
    self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
        padding=1)
    self.bn8 = nn.BatchNorm2d(512)
    self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.relu = nn.ReLU()
    self.linear = nn.Linear(in_features=512, out_features=10)
