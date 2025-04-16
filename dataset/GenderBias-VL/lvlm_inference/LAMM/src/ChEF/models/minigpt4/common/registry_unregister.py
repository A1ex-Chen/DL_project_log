@classmethod
def unregister(cls, name):
    """Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from mmf.common.registry import registry

            config = registry.unregister("config")
        """
    return cls.mapping['state'].pop(name, None)
