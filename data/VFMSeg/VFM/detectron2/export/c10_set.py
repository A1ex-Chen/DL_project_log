def set(self, name, value):
    data_len = len(value)
    if len(self.batch_extra_fields):
        assert len(self
            ) == data_len, 'Adding a field of length {} to a Instances of length {}'.format(
            data_len, len(self))
    self.batch_extra_fields[name] = value
