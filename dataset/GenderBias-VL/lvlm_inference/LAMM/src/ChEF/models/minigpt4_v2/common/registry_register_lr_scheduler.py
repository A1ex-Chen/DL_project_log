@classmethod
def register_lr_scheduler(cls, name):
    """Register a model to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from minigpt4_v2.common.registry import registry
        """

    def wrap(lr_sched_cls):
        if name in cls.mapping['lr_scheduler_name_mapping']:
            raise KeyError("Name '{}' already registered for {}.".format(
                name, cls.mapping['lr_scheduler_name_mapping'][name]))
        cls.mapping['lr_scheduler_name_mapping'][name] = lr_sched_cls
        return lr_sched_cls
    return wrap
