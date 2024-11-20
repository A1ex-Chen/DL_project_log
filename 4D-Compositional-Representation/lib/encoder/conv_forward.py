def forward(self, x):
    x = x.transpose(1, 2)
    batch_size = x.size(0)
    net = self.conv0(x)
    net = self.conv1(self.actvn(net))
    net = self.conv2(self.actvn(net))
    net = self.conv3(self.actvn(net))
    net = self.conv4(self.actvn(net))
    net = self.conv5(self.actvn(net))
    final_dim = net.shape[1]
    net = net.view(batch_size, final_dim, -1).mean(2)
    out = self.fc_out(self.actvn(net))
    return out
