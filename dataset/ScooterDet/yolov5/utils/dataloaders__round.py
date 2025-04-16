def _round(labels):
    return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]
