def _register(obj_name, obj):
    registry_key = str(obj_name)
    if registry_key in self._registry_dict:
        raise KeyError(f'{registry_key} is already registered')
    self._registry_dict[registry_key] = obj
    self._registry_dict_block_name[registry_key] = obj_name
