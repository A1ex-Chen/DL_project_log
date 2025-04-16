def forward(self, x):
    x1 = self.branch1(x)
    x2 = self.branch2(x)
    x = torch.cat((x, x1, x2), dim=1)
    x = self.conv(x)
    return x
