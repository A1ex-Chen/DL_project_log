def __repr__(self) ->str:
    head = 'Dataset ' + self.__class__.__name__
    body = [f'Number of datapoints: {self.__len__()}',
        f'Probabilities: {self.probabilities}',
        f'stopping_strategy: {self.stopping_strategy}', f'seed: {self.seed}']
    for i, ds in enumerate(self.concat_dataset.datasets):
        body.append(f'Subset {i + 1}/{len(self.concat_dataset.datasets)}')
        body += ds.__repr__().splitlines()
    lines = [head] + [(' ' * self._repr_indent + line) for line in body]
    return '\n'.join(lines)
