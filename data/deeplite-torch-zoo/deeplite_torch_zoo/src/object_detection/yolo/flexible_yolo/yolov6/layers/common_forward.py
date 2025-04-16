def forward(self, x):
    x_1 = self.conv_1(x)
    x_1 = self.blocks(x_1)
    x_2 = self.conv_2(x)
    x = torch.cat((x_1, x_2), axis=1)
    x = self.conv_3(x)
    return x
