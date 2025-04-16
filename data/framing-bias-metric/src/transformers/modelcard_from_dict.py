@classmethod
def from_dict(cls, json_object):
    """Constructs a `ModelCard` from a Python dictionary of parameters."""
    return cls(**json_object)
