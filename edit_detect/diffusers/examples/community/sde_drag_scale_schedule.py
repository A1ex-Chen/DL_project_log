def scale_schedule(begin, end, n, length, type='linear'):
    if type == 'constant':
        return end
    elif type == 'linear':
        return begin + (end - begin) * n / length
    elif type == 'cos':
        factor = (1 - math.cos(n * math.pi / length)) / 2
        return (1 - factor) * begin + factor * end
    else:
        raise NotImplementedError(type)
