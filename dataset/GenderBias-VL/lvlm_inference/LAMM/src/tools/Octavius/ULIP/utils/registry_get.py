def get(self, key):
    """Get the registry record.
        Args:
            key (str): The class name in string format.
        Returns:
            class: The corresponding class.
        """
    scope, real_key = self.split_scope_key(key)
    if scope is None or scope == self._scope:
        if real_key in self._module_dict:
            return self._module_dict[real_key]
    elif scope in self._children:
        return self._children[scope].get(real_key)
    else:
        parent = self.parent
        while parent.parent is not None:
            parent = parent.parent
        return parent.get(key)
