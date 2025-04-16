def get_function(name):
    mapping = {}
    mapping['mse'] = torch.nn.MSELoss()
    mapping['binary_crossentropy'] = torch.nn.BCELoss()
    mapping['categorical_crossentropy'] = torch.nn.CrossEntropyLoss()
    mapping['smoothL1'] = torch.nn.SmoothL1Loss()
    mapped = mapping.get(name)
    if not mapped:
        raise Exception('No pytorch function found for "{}"'.format(name))
    return mapped
