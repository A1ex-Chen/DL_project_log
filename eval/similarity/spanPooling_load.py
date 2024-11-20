@staticmethod
def load(input_path):
    with open(os.path.join(input_path, 'config.json')) as fIn:
        config = json.load(fIn)
    return Pooling(**config)
