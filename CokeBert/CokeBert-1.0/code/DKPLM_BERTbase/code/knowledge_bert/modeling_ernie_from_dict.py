@classmethod
def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size_or_config_json_file=-1)
    for key, value in json_object.items():
        config.__dict__[key] = value
    return config
