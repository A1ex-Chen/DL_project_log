def get_crop_shape(target, refer):
    """
    https://www.kaggle.com/cjansen/u-net-in-keras
    """
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert cw >= 0
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert ch >= 0
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)
    return (ch1, ch2), (cw1, cw2)
