def __init__(self, input_channels=1, output=10):
    super(LeNet5, self).__init__()
    self.conv1 = nn.Conv2d(input_channels, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(256, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, output)
