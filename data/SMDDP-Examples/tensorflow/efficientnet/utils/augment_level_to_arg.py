def level_to_arg(cutout_const: float, translate_const: float):
    """Creates a dict mapping image operation names to their arguments."""
    no_arg = lambda level: ()
    posterize_arg = lambda level: _mult_to_arg(level, 4)
    solarize_arg = lambda level: _mult_to_arg(level, 256)
    solarize_add_arg = lambda level: _mult_to_arg(level, 110)
    cutout_arg = lambda level: _mult_to_arg(level, cutout_const)
    translate_arg = lambda level: _translate_level_to_arg(level,
        translate_const)
    args = {'AutoContrast': no_arg, 'Equalize': no_arg, 'Invert': no_arg,
        'Rotate': _rotate_level_to_arg, 'Posterize': posterize_arg,
        'Solarize': solarize_arg, 'SolarizeAdd': solarize_add_arg, 'Color':
        _enhance_level_to_arg, 'Contrast': _enhance_level_to_arg,
        'Brightness': _enhance_level_to_arg, 'Sharpness':
        _enhance_level_to_arg, 'ShearX': _shear_level_to_arg, 'ShearY':
        _shear_level_to_arg, 'Cutout': cutout_arg, 'TranslateX':
        translate_arg, 'TranslateY': translate_arg}
    return args
