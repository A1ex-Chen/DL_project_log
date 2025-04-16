def get_function(name):
    mapping = {}
    mapped = mapping.get(name)
    if not mapped:
        raise Exception('No keras function found for "{}"'.format(name))
    return mapped
