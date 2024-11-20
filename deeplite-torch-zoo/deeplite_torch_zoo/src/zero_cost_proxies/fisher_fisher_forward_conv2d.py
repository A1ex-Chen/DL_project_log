def fisher_forward_conv2d(self, x):
    x = nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.
        padding, self.dilation, self.groups)
    self.act = self.dummy(x)
    return self.act
