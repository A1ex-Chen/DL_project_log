@classmethod
def register_processor(cls, name):
    """Register a processor to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from lavis.common.registry import registry
        """

    def wrap(processor_cls):
        from ..processors import BaseProcessor
        assert issubclass(processor_cls, BaseProcessor
            ), 'All processors must inherit BaseProcessor class'
        if name in cls.mapping['processor_name_mapping']:
            raise KeyError("Name '{}' already registered for {}.".format(
                name, cls.mapping['processor_name_mapping'][name]))
        cls.mapping['processor_name_mapping'][name] = processor_cls
        return processor_cls
    return wrap
