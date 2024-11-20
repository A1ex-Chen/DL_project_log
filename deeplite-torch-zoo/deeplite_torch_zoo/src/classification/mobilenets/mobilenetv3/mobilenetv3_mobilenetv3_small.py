def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [[3, 1, 16, 1, 0, 2], [3, 4.5, 24, 0, 0, 2], [3, 3.67, 24, 0, 0,
        1], [5, 4, 40, 1, 1, 2], [5, 6, 40, 1, 1, 1], [5, 6, 40, 1, 1, 1],
        [5, 3, 48, 1, 1, 1], [5, 3, 48, 1, 1, 1], [5, 6, 96, 1, 1, 2], [5, 
        6, 96, 1, 1, 1], [5, 6, 96, 1, 1, 1]]
    return MobileNetV3(cfgs, mode='small', **kwargs)
