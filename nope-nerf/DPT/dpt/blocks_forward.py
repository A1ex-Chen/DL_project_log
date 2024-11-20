def forward(self, *xs):
    """Forward pass.

        Returns:
            tensor: output
        """
    output = xs[0]
    if len(xs) == 2:
        res = self.resConfUnit1(xs[1])
        output = self.skip_add.add(output, res)
    output = self.resConfUnit2(output)
    output = nn.functional.interpolate(output, scale_factor=2, mode=
        'bilinear', align_corners=self.align_corners)
    output = self.out_conv(output)
    return output
