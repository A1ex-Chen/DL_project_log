def forward(self, x):
    y1 = self.cv3(self.m(self.cv1(x)))
    y2 = self.cv2(x)
    return self.cv4(torch.cat((y1, y2), dim=1))
