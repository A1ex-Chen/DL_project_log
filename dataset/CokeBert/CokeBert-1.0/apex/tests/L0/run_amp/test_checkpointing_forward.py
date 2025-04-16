def forward(self, x):
    x = x * self.param
    x = F.relu(self.conv1(x))
    x = self.bn1(x)
    return x
