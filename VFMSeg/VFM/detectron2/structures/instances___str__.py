def __str__(self) ->str:
    s = self.__class__.__name__ + '('
    s += 'num_instances={}, '.format(len(self))
    s += 'image_height={}, '.format(self._image_size[0])
    s += 'image_width={}, '.format(self._image_size[1])
    s += 'fields=[{}])'.format(', '.join(f'{k}: {v}' for k, v in self.
        _fields.items()))
    return s
