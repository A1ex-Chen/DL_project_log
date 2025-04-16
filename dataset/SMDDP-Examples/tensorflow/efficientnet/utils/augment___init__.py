def __init__(self, num_layers: int=2, magnitude: float=10.0, cutout_const:
    float=40.0, translate_const: float=100.0):
    """Applies the RandAugment policy to images.

    Args:
      num_layers: Integer, the number of augmentation transformations to apply
        sequentially to an image. Represented as (N) in the paper. Usually best
        values will be in the range [1, 3].
      magnitude: Integer, shared magnitude across all augmentation operations.
        Represented as (M) in the paper. Usually best values are in the range
        [5, 10].
      cutout_const: multiplier for applying cutout.
      translate_const: multiplier for applying translation.
    """
    super(RandAugment, self).__init__()
    self.num_layers = num_layers
    self.magnitude = float(magnitude)
    self.cutout_const = float(cutout_const)
    self.translate_const = float(translate_const)
    self.available_ops = ['AutoContrast', 'Equalize', 'Invert', 'Rotate',
        'Posterize', 'Solarize', 'Color', 'Contrast', 'Brightness',
        'Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY',
        'Cutout', 'SolarizeAdd']
