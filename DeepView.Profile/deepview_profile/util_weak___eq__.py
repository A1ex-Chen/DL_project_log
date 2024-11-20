def __eq__(self, other):
    if not isinstance(other, Mapping):
        return NotImplemented
    return {id(k): v for k, v in self.items()} == {id(k): v for k, v in
        other.items()}
