def forward(self, x, x_skip=None):
    x_res = x
    x = self.norm(self.depthwise(x))
    if x_skip is not None:
        x = torch.cat([x, x_skip], dim=1)
    x = self.channelwise(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    return x + x_res
