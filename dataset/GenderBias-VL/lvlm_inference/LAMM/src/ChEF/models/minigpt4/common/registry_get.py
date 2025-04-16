@classmethod
def get(cls, name, default=None, no_warning=False):
    """Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retrieved.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for MMF's
                               internal operations. Default: False
        """
    original_name = name
    name = name.split('.')
    value = cls.mapping['state']
    for subname in name:
        value = value.get(subname, default)
        if value is default:
            break
    if 'writer' in cls.mapping['state'
        ] and value == default and no_warning is False:
        cls.mapping['state']['writer'].warning(
            'Key {} is not present in registry, returning default value of {}'
            .format(original_name, default))
    return value
