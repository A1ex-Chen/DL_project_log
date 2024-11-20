def to_dict(self):
    """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support).
        """
    d = dataclasses.asdict(self)
    for k, v in d.items():
        if isinstance(v, Enum):
            d[k] = v.value
    return d
