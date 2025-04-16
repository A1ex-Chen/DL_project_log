def _shrink_level_to_arg(level: float):
    """Converts level to ratio by which we shrink the image content."""
    if level == 0:
        return 1.0,
    level = 2.0 / (_MAX_LEVEL / level) + 0.9
    return level,
