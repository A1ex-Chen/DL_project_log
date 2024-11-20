@staticmethod
def hex2rgb(h):
    return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
