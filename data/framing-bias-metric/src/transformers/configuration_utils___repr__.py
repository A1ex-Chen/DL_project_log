def __repr__(self):
    return '{} {}'.format(self.__class__.__name__, self.to_json_string())
