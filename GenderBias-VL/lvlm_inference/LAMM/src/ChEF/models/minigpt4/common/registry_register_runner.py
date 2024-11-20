@classmethod
def register_runner(cls, name):
    """Register a model to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from minigpt4.common.registry import registry
        """

    def wrap(runner_cls):
        if name in cls.mapping['runner_name_mapping']:
            raise KeyError("Name '{}' already registered for {}.".format(
                name, cls.mapping['runner_name_mapping'][name]))
        cls.mapping['runner_name_mapping'][name] = runner_cls
        return runner_cls
    return wrap
