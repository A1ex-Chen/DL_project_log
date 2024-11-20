def forward(self, x):
    if isinstance(x, list):
        x = torch.cat(x, 1)
    return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
