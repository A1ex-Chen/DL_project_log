def _load_file(self, filename):
    if filename.endswith('.pkl'):
        with PathManager.open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        if 'model' in data and '__author__' in data:
            self.logger.info("Reading a file from '{}'".format(data[
                '__author__']))
            return data
        else:
            if 'blobs' in data:
                data = data['blobs']
            data = {k: v for k, v in data.items() if not k.endswith(
                '_momentum')}
            return {'model': data, '__author__': 'Caffe2',
                'matching_heuristics': True}
    elif filename.endswith('.pyth'):
        with PathManager.open(filename, 'rb') as f:
            data = torch.load(f)
        assert 'model_state' in data, f"Cannot load .pyth file {filename}; pycls checkpoints must contain 'model_state'."
        model_state = {k: v for k, v in data['model_state'].items() if not
            k.endswith('num_batches_tracked')}
        return {'model': model_state, '__author__': 'pycls',
            'matching_heuristics': True}
    loaded = super()._load_file(filename)
    if 'model' not in loaded:
        loaded = {'model': loaded}
    return loaded
