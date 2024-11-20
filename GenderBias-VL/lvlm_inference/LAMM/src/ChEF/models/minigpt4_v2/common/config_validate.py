def validate(self, config=None):
    """
        Convert yaml config (dict-like) to list, required by argparse.
        """
    for k, v in config.items():
        assert k in self.arguments, f'{k} is not a valid argument. Support arguments are {self.format_arguments()}.'
        if self.arguments[k].type is not None:
            try:
                self.arguments[k].val = self.arguments[k].type(v)
            except ValueError:
                raise ValueError(
                    f'{k} is not a valid {self.arguments[k].type}.')
        if self.arguments[k].choices is not None:
            assert v in self.arguments[k
                ].choices, f'{k} must be one of {self.arguments[k].choices}.'
    return config
