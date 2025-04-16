@classmethod
def list_tasks(cls):
    return sorted(cls.mapping['task_name_mapping'].keys())
