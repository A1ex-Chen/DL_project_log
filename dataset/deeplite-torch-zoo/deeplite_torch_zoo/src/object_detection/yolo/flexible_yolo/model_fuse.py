def fuse(self, verbose=False):
    LOGGER.info('Fusing layers... ')
    for m in self.modules():
        if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)
            delattr(m, 'bn')
            m.forward = m.forward_fuse
        if isinstance(m, RepConv):
            m.fuse_repvgg_block()
    self.info()
    self._is_fused = True
    return self
