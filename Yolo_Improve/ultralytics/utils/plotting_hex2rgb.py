@staticmethod
def hex2rgb(h):
    """Converts hex color codes to RGB values (i.e. default PIL order)."""
    return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
