def __repr__(self) ->str:
    head = 'Dataset ' + self.__class__.__name__
    body = [f'Number of datapoints: {self.__len__()}',
        f'ann file: {self.filename}']
    if self.image_folder is not None:
        body.append(f'image folder: {self.image_folder}')
    body += self.extra_repr().splitlines()
    lines = [head] + [(' ' * self._repr_indent + line) for line in body]
    return '\n'.join(lines)
