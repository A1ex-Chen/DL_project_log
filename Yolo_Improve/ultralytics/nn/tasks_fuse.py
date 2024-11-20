def fuse(self, verbose=True):
    """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
    if not self.is_fused():
        for m in self.model.modules():
            if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, 'bn'):
                if isinstance(m, Conv2):
                    m.fuse_convs()
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
            if isinstance(m, ConvTranspose) and hasattr(m, 'bn'):
                m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
            if isinstance(m, RepConv):
                m.fuse_convs()
                m.forward = m.forward_fuse
            if isinstance(m, RepVGGDW):
                m.fuse()
                m.forward = m.forward_fuse
        self.info(verbose=verbose)
    return self
