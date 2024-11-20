def forward(self, x):
    x = self.model(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x
