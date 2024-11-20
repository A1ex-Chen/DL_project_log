def set(self, name: str, value: Any) ->None:
    """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
    data_len = len(value)
    if len(self._fields):
        assert len(self
            ) == data_len, 'Adding a field of length {} to a Instances of length {}'.format(
            data_len, len(self))
    self._fields[name] = value
