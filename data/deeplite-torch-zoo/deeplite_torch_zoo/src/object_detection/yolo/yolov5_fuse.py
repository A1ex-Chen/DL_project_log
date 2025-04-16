def fuse(self, verbose=False):
    LOGGER.info('Fusing layers... ')
    for m in self.model.modules():
        if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
            if verbose:
                LOGGER.info(f'Fusing BN into Conv for module {m}')
            m.conv = fuse_conv_and_bn(m.conv, m.bn)
            delattr(m, 'bn')
            m.forward = m.forward_fuse
        if isinstance(m, RepConv) or isinstance(m, MobileOneBlock):
            if verbose:
                LOGGER.info(f'Fusing RepVGG-style module {m}')
            m.fuse()
    self.info()
    self._is_fused = True
    return self
