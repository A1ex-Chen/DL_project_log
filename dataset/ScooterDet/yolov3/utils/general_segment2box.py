def segment2box(segment, width=640, height=640):
    x, y = segment.T
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x
        ) else np.zeros((1, 4))
