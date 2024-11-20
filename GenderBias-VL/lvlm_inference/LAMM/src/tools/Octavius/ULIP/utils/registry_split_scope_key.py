@staticmethod
def split_scope_key(key):
    """Split scope and key.
        The first scope will be split from key.
        Examples:
            >>> Registry.split_scope_key('mmdet.ResNet')
            'mmdet', 'ResNet'
            >>> Registry.split_scope_key('ResNet')
            None, 'ResNet'
        Return:
            scope (str, None): The first scope.
            key (str): The remaining key.
        """
    split_index = key.find('.')
    if split_index != -1:
        return key[:split_index], key[split_index + 1:]
    else:
        return None, key
