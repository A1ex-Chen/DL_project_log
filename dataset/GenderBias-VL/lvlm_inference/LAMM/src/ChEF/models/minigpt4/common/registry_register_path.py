@classmethod
def register_path(cls, name, path):
    """Register a path to registry with key 'name'

        Args:
            name: Key with which the path will be registered.

        Usage:

            from minigpt4.common.registry import registry
        """
    assert isinstance(path, str), 'All path must be str.'
    if name in cls.mapping['paths']:
        raise KeyError("Name '{}' already registered.".format(name))
    cls.mapping['paths'][name] = path
