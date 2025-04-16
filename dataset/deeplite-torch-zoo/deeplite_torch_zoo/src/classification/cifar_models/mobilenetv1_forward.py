def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.layers(out)
    out = F.avg_pool2d(out, 2)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out
