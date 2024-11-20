def __init__(self, p1, operation1, magnitude_idx1, p2, operation2,
    magnitude_idx2, fillcolor=(128, 128, 128)):
    ranges = {'shearX': np.linspace(0, 0.3, 10), 'shearY': np.linspace(0, 
        0.3, 10), 'translateX': np.linspace(0, 150 / 331, 10), 'translateY':
        np.linspace(0, 150 / 331, 10), 'rotate': np.linspace(0, 30, 10),
        'color': np.linspace(0.0, 0.9, 10), 'posterize': np.round(np.
        linspace(8, 4, 10), 0).astype(np.int), 'solarize': np.linspace(256,
        0, 10), 'contrast': np.linspace(0.0, 0.9, 10), 'sharpness': np.
        linspace(0.0, 0.9, 10), 'brightness': np.linspace(0.0, 0.9, 10),
        'autocontrast': [0] * 10, 'equalize': [0] * 10, 'invert': [0] * 10}

    def rotate_with_fill(img, magnitude):
        rot = img.convert('RGBA').rotate(magnitude)
        return Image.composite(rot, Image.new('RGBA', rot.size, (128,) * 4),
            rot).convert(img.mode)
    func = {'shearX': lambda img, magnitude: img.transform(img.size, Image.
        AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0), Image.
        BICUBIC, fillcolor=fillcolor), 'shearY': lambda img, magnitude: img
        .transform(img.size, Image.AFFINE, (1, 0, 0, magnitude * random.
        choice([-1, 1]), 1, 0), Image.BICUBIC, fillcolor=fillcolor),
        'translateX': lambda img, magnitude: img.transform(img.size, Image.
        AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0,
        1, 0), fillcolor=fillcolor), 'translateY': lambda img, magnitude:
        img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude *
        img.size[1] * random.choice([-1, 1])), fillcolor=fillcolor),
        'rotate': rotate_with_fill, 'color': lambda img, magnitude:
        ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 
        1])), 'posterize': ImageOps.posterize, 'solarize': ImageOps.
        solarize, 'contrast': lambda img, magnitude: ImageEnhance.Contrast(
        img).enhance(1 + magnitude * random.choice([-1, 1])), 'sharpness': 
        lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(1 + 
        magnitude * random.choice([-1, 1])), 'brightness': lambda img,
        magnitude: ImageEnhance.Brightness(img).enhance(1 + magnitude *
        random.choice([-1, 1])), 'autocontrast': lambda img, magnitude:
        ImageOps.autocontrast(img), 'equalize': lambda img, magnitude:
        ImageOps.equalize(img), 'invert': lambda img, magnitude: ImageOps.
        invert(img)}
    self.p1 = p1
    self.operation1 = func[operation1]
    self.magnitude1 = ranges[operation1][magnitude_idx1]
    self.p2 = p2
    self.operation2 = func[operation2]
    self.magnitude2 = ranges[operation2][magnitude_idx2]
