def get_metric(self, name):
    if name == 'map':
        return get_map
    elif name == 'acc':
        return get_acc
    elif name == 'mauc':
        return get_mauc
    else:
        raise ValueError(
            f'the metric should be at least one of [map, acc, mauc]')
