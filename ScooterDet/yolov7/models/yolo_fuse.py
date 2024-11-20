def fuse(self):
    print('Fusing layers... ')
    for m in self.model.modules():
        if isinstance(m, RepConv):
            m.fuse_repvgg_block()
        elif isinstance(m, RepConv_OREPA):
            m.switch_to_deploy()
        elif type(m) is Conv and hasattr(m, 'bn'):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)
            delattr(m, 'bn')
            m.forward = m.fuseforward
        elif isinstance(m, (IDetect, IAuxDetect)):
            m.fuse()
            m.forward = m.fuseforward
    self.info()
    return self
