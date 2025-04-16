def _translate_level_to_arg(level: float, translate_const: float):
    level = level / _MAX_LEVEL * float(translate_const)
    level = _randomly_negate_tensor(level)
    return level,
