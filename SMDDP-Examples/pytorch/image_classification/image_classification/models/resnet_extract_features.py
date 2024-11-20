def extract_features(self, x, layers=None):
    if layers is None:
        layers = [f'layer{i + 1}' for i in range(self.num_layers)] + [
            'classifier']
    run = [f'layer{i + 1}' for i in range(self.num_layers) if 'classifier' in
        layers or any([(f'layer{j + 1}' in layers) for j in range(i, self.
        num_layers)])]
    output = {}
    x = self.stem(x)
    for l in run:
        fn = getattr(self, l)
        x = fn(x)
        if l in layers:
            output[l] = x
    if 'classifier' in layers:
        output['classifier'] = self.classifier(x)
    return output
