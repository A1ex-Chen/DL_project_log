@classmethod
def register_builder(cls, name):
    """Register a dataset builder to registry with key 'name'

        Args:
            name: Key with which the builder will be registered.

        Usage:

            from minigpt4_v2.common.registry import registry
            from minigpt4_v2.datasets.base_dataset_builder import BaseDatasetBuilder
        """

    def wrap(builder_cls):
        from ..datasets.builders.base_dataset_builder import BaseDatasetBuilder
        assert issubclass(builder_cls, BaseDatasetBuilder
            ), 'All builders must inherit BaseDatasetBuilder class, found {}'.format(
            builder_cls)
        if name in cls.mapping['builder_name_mapping']:
            raise KeyError("Name '{}' already registered for {}.".format(
                name, cls.mapping['builder_name_mapping'][name]))
        cls.mapping['builder_name_mapping'][name] = builder_cls
        return builder_cls
    return wrap
