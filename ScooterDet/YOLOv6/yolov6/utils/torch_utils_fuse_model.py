def fuse_model(model):
    """Fuse convolution and batchnorm layers of the model."""
    from yolov6.layers.common import ConvModule
    for m in model.modules():
        if type(m) is ConvModule and hasattr(m, 'bn'):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)
            delattr(m, 'bn')
            m.forward = m.forward_fuse
    return model
