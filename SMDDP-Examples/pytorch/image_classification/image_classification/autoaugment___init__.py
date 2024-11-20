def __init__(self):
    fillcolor = 128, 128, 128
    self.ranges = {'shearX': np.linspace(0, 0.3, 11), 'shearY': np.linspace
        (0, 0.3, 11), 'translateX': np.linspace(0, 250, 11), 'translateY':
        np.linspace(0, 250, 11), 'rotate': np.linspace(0, 30, 11), 'color':
        np.linspace(0.1, 1.9, 11), 'posterize': np.round(np.linspace(0, 4, 
        11), 0).astype(np.int), 'solarize': np.linspace(0, 256, 11),
        'solarizeadd': np.linspace(0, 110, 11), 'contrast': np.linspace(0.1,
        1.9, 11), 'sharpness': np.linspace(0.1, 1.9, 11), 'brightness': np.
        linspace(0.1, 1.9, 11), 'autocontrast': [0] * 10, 'equalize': [0] *
        10, 'invert': [0] * 10}

    def rotate_with_fill(img, magnitude):
        magnitude *= random.choice([-1, 1])
        rot = img.convert('RGBA').rotate(magnitude)
        return Image.composite(rot, Image.new('RGBA', rot.size, (128,) * 4),
            rot).convert(img.mode)

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
    self.operations = {'shearX': lambda img, magnitude: img.transform(img.
        size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1,
        0), Image.BICUBIC, fillcolor=fillcolor), 'shearY': lambda img,
        magnitude: img.transform(img.size, Image.AFFINE, (1, 0, 0, 
        magnitude * random.choice([-1, 1]), 1, 0), Image.BICUBIC, fillcolor
        =fillcolor), 'translateX': lambda img, magnitude: img.transform(img
        .size, Image.AFFINE, (1, 0, magnitude * random.choice([-1, 1]), 0, 
        1, 0), fillcolor=fillcolor), 'translateY': lambda img, magnitude:
        img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude *
        random.choice([-1, 1])), fillcolor=fillcolor), 'rotate': lambda img,
        magnitude: rotate_with_fill(img, magnitude), 'color': lambda img,
        magnitude: ImageEnhance.Color(img).enhance(magnitude), 'posterize':
        lambda img, magnitude: ImageOps.posterize(img, magnitude),
        'solarize': lambda img, magnitude: ImageOps.solarize(img, magnitude
        ), 'solarizeadd': lambda img, magnitude: solarize_add(img,
        magnitude), 'contrast': lambda img, magnitude: ImageEnhance.
        Contrast(img).enhance(magnitude), 'sharpness': lambda img,
        magnitude: ImageEnhance.Sharpness(img).enhance(magnitude),
        'brightness': lambda img, magnitude: ImageEnhance.Brightness(img).
        enhance(magnitude), 'autocontrast': lambda img, _: ImageOps.
        autocontrast(img), 'equalize': lambda img, _: ImageOps.equalize(img
        ), 'invert': lambda img, _: ImageOps.invert(img)}
