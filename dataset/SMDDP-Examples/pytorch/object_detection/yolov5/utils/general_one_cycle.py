def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: (1 - math.cos(x * math.pi / steps)) / 2 * (y2 - y1) + y1
