def forward(self, x):
    y = self.cv1(x)
    if self.single_conv:
        return y
    res = torch.cat([y, self.cv2(y)], 1
        ) if not self.residual else x + torch.cat([y, self.cv2(y)], 1)
    if self.dfc:
        res = res * self.dfc(x)
    return res
