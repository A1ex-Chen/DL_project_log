@classmethod
def get_runner_class(cls, name):
    return cls.mapping['runner_name_mapping'].get(name, None)
