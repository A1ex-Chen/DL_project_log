def __init__(self):
    super(LeNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5)
    self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5)
    self.dense1 = nn.Linear(in_features=1250, out_features=500)
    self.dense2 = nn.Linear(in_features=500, out_features=10)
    self.tanh = nn.Tanh()
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.softmax = nn.LogSoftmax(dim=1)
