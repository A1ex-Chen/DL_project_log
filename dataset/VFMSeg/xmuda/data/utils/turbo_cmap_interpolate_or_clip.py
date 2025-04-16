def interpolate_or_clip(colormap, x):
    if x < 0.0:
        return [0.0, 0.0, 0.0]
    elif x > 1.0:
        return [1.0, 1.0, 1.0]
    else:
        return interpolate(colormap, x)
