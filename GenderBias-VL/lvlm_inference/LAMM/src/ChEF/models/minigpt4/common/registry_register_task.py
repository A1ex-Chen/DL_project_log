@classmethod
def register_task(cls, name):
    """Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from minigpt4.common.registry import registry
        """

    def wrap(task_cls):
        from minigpt4.tasks.base_task import BaseTask
        assert issubclass(task_cls, BaseTask
            ), 'All tasks must inherit BaseTask class'
        if name in cls.mapping['task_name_mapping']:
            raise KeyError("Name '{}' already registered for {}.".format(
                name, cls.mapping['task_name_mapping'][name]))
        cls.mapping['task_name_mapping'][name] = task_cls
        return task_cls
    return wrap
