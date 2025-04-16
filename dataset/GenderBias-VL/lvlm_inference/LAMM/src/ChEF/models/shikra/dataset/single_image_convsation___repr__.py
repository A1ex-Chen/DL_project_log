def __repr__(self) ->str:
    head = 'Dataset ' + self.__class__.__name__
    body = [f'Number of datapoints: {self.__len__()}']
    body += self.dataset.__repr__().splitlines()
    lines = [head] + [(' ' * self._repr_indent + line) for line in body]
    return '\n'.join(lines)
