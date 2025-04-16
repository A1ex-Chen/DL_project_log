def get_block(mode):
    if mode == 'repvgg':
        return RepVGGBlock
    elif mode == 'qarepvgg':
        return QARepVGGBlock
    elif mode == 'qarepvggv2':
        return QARepVGGBlockV2
    elif mode == 'hyper_search':
        return LinearAddBlock
    elif mode == 'repopt':
        return RealVGGBlock
    elif mode == 'conv_relu':
        return ConvBNReLU
    elif mode == 'conv_silu':
        return ConvBNSiLU
    else:
        raise NotImplementedError('Undefied Repblock choice for mode {}'.
            format(mode))
