def forward(self, x):
    shortcut = x
    if self.conv_exp is not None:
        x = self.conv_exp(x)
    x = self.conv_dw(x)
    x = self.se(x)
    x = self.act_dw(x)
    x = self.conv_pwl(x)
    x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.
        in_channels:]], dim=1)
    return x
