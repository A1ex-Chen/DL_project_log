def forward(self, x):
    x1 = self.cv4(self.cv3(self.cv1(x)))
    y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
    y2 = self.cv2(x)
    return self.cv7(torch.cat((y1, y2), dim=1))
