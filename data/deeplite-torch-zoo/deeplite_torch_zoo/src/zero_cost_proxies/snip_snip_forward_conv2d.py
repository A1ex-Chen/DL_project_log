def snip_forward_conv2d(self, x):
    return nn.functional.conv2d(x, self.weight * self.weight_mask, self.
        bias, self.stride, self.padding, self.dilation, self.groups)
