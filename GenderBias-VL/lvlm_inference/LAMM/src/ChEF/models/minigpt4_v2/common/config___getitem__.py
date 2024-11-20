def __getitem__(self, key):
    assert self.parsed_args is not None, 'No arguments parsed yet.'
    return self.parsed_args[key]
