def max_point_str(val, max_point):
    positive = bool(val >= 0.0)
    val = np.abs(val)
    if val == 0:
        point = 1
    else:
        point = max(int(np.log10(val)), 0) + 1
    fmt = '{:.' + str(max(max_point - point, 0)) + 'f}'
    if positive is True:
        return fmt.format(val)
    else:
        return fmt.format(-val)
