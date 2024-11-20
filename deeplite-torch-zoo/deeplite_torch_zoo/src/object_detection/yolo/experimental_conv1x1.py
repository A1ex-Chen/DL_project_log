@staticmethod
def conv1x1(i, o, kernel_size=1, stride=1, padding=0, bias=False):
    return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias)
