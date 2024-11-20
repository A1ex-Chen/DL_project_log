@classmethod
def from_dict(cls, json_object):
    """Constructs a `Config` from a Python dictionary of parameters."""
    config = cls(vocab_size_or_config_json_file=-1)
    for key, value in json_object.items():
        setattr(config, key, value)
    return config
