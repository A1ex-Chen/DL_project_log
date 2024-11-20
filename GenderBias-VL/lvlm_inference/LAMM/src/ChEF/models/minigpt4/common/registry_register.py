@classmethod
def register(cls, name, obj):
    """Register an item to registry with key 'name'

        Args:
            name: Key with which the item will be registered.

        Usage::

            from minigpt4.common.registry import registry

            registry.register("config", {})
        """
    path = name.split('.')
    current = cls.mapping['state']
    for part in path[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[path[-1]] = obj
