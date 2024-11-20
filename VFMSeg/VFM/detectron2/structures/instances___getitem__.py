def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) ->'Instances':
    """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
    if type(item) == int:
        if item >= len(self) or item < -len(self):
            raise IndexError('Instances index out of range!')
        else:
            item = slice(item, None, len(self))
    ret = Instances(self._image_size)
    for k, v in self._fields.items():
        ret.set(k, v[item])
    return ret
