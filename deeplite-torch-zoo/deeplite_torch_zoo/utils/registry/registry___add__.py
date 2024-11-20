def __add__(self, other_registry):
    """
        Adding two objects of type Registry results in merging registry dicts
        """
    res = type(self)(self._name)
    if self._registry_dict.keys() & other_registry.registry_dict.keys():
        raise ValueError('Trying to add two registries with overlapping keys')
    res._registry_dict.update(self._registry_dict)
    res._registry_dict.update(other_registry.registry_dict)
    return res
