def _register(obj_name, obj):
    if obj_name in self._registry_dict:
        raise KeyError(
            f'{obj_name} is already registered in the evaluator wrapper registry'
            )
    self._registry_dict[obj_name] = obj
