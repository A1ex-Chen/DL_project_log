def register(self, task_type, model_type=None, dataset_type=None):

    def _register(obj_name, obj):
        if obj_name in self._registry_dict:
            raise KeyError(
                f'{obj_name} is already registered in the evaluator wrapper registry'
                )
        self._registry_dict[obj_name] = obj

    def wrap(obj):
        cls_name = task_type
        if cls_name is None:
            cls_name = obj.__name__
        cls_name = self._registry_key(task_type=cls_name, model_type=
            model_type, dataset_type=dataset_type)
        _register(cls_name, obj)
        return obj
    return wrap
