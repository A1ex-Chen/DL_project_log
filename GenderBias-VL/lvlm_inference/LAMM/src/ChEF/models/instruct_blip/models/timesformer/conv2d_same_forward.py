def forward(self, x):
    return conv2d_same(x, self.weight, self.bias, self.stride, self.padding,
        self.dilation, self.groups)
