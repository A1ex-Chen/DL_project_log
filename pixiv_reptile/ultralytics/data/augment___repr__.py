def __repr__(self):
    """Returns a string representation of the object."""
    return (
        f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"
        )
