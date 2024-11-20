def forward(self, input):
    out = None
    input_shape = numpy.shape(input)[2:]
    for f in self.features:
        x = f(input)
        x = F.interpolate(x, input_shape, mode='bilinear', align_corners=
            self.align_corners)
        if out is None:
            out = x
        else:
            out += x
    out = self.final_conv(out)
    return out
