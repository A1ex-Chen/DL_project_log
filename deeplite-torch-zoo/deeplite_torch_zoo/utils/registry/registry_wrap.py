def wrap(obj):
    cls_name = task_type
    if cls_name is None:
        cls_name = obj.__name__
    cls_name = self._registry_key(task_type=cls_name, model_type=model_type,
        dataset_type=dataset_type)
    _register(cls_name, obj)
    return obj