def solarize_add(image, addition=0, threshold=128):
    lut = []
    for i in range(256):
        if i < threshold:
            res = i + addition if i + addition <= 255 else 255
            res = res if res >= 0 else 0
            lut.append(res)
        else:
            lut.append(i)
    from PIL.ImageOps import _lut
    return _lut(image, lut)
