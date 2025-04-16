@classmethod
def get_task_class(cls, name):
    return cls.mapping['task_name_mapping'].get(name, None)
