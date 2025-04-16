def extra_repr(self):
    tmpstr = 'in_channels=' + str(self.in_channels)
    tmpstr += ', out_channels=' + str(self.out_channels)
    tmpstr += ', kernel_size=' + str(self.kernel_size)
    tmpstr += ', stride=' + str(self.stride)
    tmpstr += ', padding=' + str(self.padding)
    tmpstr += ', dilation=' + str(self.dilation)
    tmpstr += ', groups=' + str(self.groups)
    tmpstr += ', deformable_groups=' + str(self.deformable_groups)
    tmpstr += ', bias=' + str(self.with_bias)
    return tmpstr
