def extract_features(self, x, layers=None):
    if layers is None:
        layers = [f'layer{i + 1}' for i in range(self.num_layers)]
    run = [f'layer{i + 1}' for i in range(self.num_layers) if 'classifier' in
        layers or 'features' in layers or any([(f'layer{j + 1}' in layers) for
        j in range(i, self.num_layers)])]
    if 'features' in layers or 'classifier' in layers:
        run.append('features')
    if 'classifier' in layers:
        run.append('classifier')
    output = {}
    x = self.stem(x)
    for l in run:
        fn = getattr(self, l)
        x = fn(x)
        if l in layers:
            output[l] = x
    return output
