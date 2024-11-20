def fuse(self):
    LOGGER.info('Fusing layers... ')
    for m in self.model.modules():
        if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)
            delattr(m, 'bn')
            m.forward = m.forward_fuse
    self.info()
    return self
