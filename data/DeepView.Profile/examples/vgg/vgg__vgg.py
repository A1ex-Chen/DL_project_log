def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    return VGG(cfgs[cfg], batch_norm=batch_norm, **kwargs)
