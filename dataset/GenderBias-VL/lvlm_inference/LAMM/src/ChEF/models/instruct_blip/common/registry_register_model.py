@classmethod
def register_model(cls, name):
    """Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from lavis.common.registry import registry
        """

    def wrap(model_cls):
        from ..models import BaseModel
        assert issubclass(model_cls, BaseModel
            ), 'All models must inherit BaseModel class'
        if name in cls.mapping['model_name_mapping']:
            raise KeyError("Name '{}' already registered for {}.".format(
                name, cls.mapping['model_name_mapping'][name]))
        cls.mapping['model_name_mapping'][name] = model_cls
        return model_cls
    return wrap
