def _shear_level_to_arg(level: float):
    level = level / _MAX_LEVEL * 0.3
    level = _randomly_negate_tensor(level)
    return level,
