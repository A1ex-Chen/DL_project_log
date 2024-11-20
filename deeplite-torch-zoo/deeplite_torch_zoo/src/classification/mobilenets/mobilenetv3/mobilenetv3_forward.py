def forward(self, x):
    x = self.features(x)
    x = self.conv(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x
