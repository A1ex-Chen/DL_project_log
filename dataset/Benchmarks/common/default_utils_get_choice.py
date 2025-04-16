def get_choice(name):
    """Maps name string to the right type of argument"""
    mapping = {}
    mapping['f16'] = np.float16
    mapping['f32'] = np.float32
    mapping['f64'] = np.float64
    mapped = mapping.get(name)
    if not mapped:
        raise Exception('No mapping found for "{}"'.format(name))
    return mapped
