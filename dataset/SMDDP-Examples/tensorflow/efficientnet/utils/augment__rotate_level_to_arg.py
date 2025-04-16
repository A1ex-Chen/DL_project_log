def _rotate_level_to_arg(level: float):
    level = level / _MAX_LEVEL * 30.0
    level = _randomly_negate_tensor(level)
    return level,
