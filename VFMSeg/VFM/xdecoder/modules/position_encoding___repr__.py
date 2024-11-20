def __repr__(self, _repr_indent=4):
    head = 'Positional encoding ' + self.__class__.__name__
    body = ['num_pos_feats: {}'.format(self.num_pos_feats),
        'temperature: {}'.format(self.temperature), 'normalize: {}'.format(
        self.normalize), 'scale: {}'.format(self.scale)]
    lines = [head] + [(' ' * _repr_indent + line) for line in body]
    return '\n'.join(lines)
