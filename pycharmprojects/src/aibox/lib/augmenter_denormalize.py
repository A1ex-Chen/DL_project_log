@staticmethod
def denormalize(name: str, normalized_min_max: Tuple[float, float]) ->Tuple[
    float, float]:
    if name == 'crop':
        assert 0 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
        func = lambda x: x / 5
    elif name == 'zoom':
        assert -1 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
        func = lambda x: (x + 2) / 2
    elif name == 'scale':
        assert -1 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
        func = lambda x: (x + 2) / 2
    elif name == 'translate':
        assert -1 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
        func = lambda x: x * 0.3
    elif name == 'rotate':
        assert -1 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
        func = lambda x: x * 45
    elif name == 'shear':
        assert -1 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
        func = lambda x: x * 30
    elif name == 'blur':
        assert 0 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
        func = lambda x: x * 10
    elif name == 'sharpen':
        assert 0 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
        func = lambda x: x
    elif name == 'color':
        assert -1 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
        func = lambda x: int(x * 50)
    elif name == 'brightness':
        assert -1 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
        func = lambda x: int(x * 50)
    elif name == 'grayscale':
        assert 0 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
        func = lambda x: x
    elif name == 'contrast':
        assert -1 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
        func = lambda x: (x + 2) / 2
    elif name == 'noise':
        assert 0 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
        func = lambda x: x / 2
    else:
        raise ValueError
    return func(normalized_min_max[0]), func(normalized_min_max[1])
