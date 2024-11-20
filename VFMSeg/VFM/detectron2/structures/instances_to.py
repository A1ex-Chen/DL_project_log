def to(self, *args: Any, **kwargs: Any) ->'Instances':
    """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
    ret = Instances(self._image_size)
    for k, v in self._fields.items():
        if hasattr(v, 'to'):
            v = v.to(*args, **kwargs)
        ret.set(k, v)
    return ret
