def interpolate(colormap, x):
    x = max(0.0, min(1.0, x))
    a = int(x * 255.0)
    b = min(255, a + 1)
    f = x * 255.0 - a
    return [colormap[a][0] + (colormap[b][0] - colormap[a][0]) * f, 
        colormap[a][1] + (colormap[b][1] - colormap[a][1]) * f, colormap[a]
        [2] + (colormap[b][2] - colormap[a][2]) * f]
