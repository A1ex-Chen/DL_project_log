def arcball_map_to_sphere(point, center, radius):
    """Return unit sphere coordinates from window coordinates."""
    v = numpy.array(((point[0] - center[0]) / radius, (center[1] - point[1]
        ) / radius, 0.0), dtype=numpy.float64)
    n = v[0] * v[0] + v[1] * v[1]
    if n > 1.0:
        v /= math.sqrt(n)
    else:
        v[2] = math.sqrt(1.0 - n)
    return v
