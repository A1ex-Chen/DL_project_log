def unpickle(file):
    """load the cifar-10 data"""
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data
