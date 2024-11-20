@classmethod
def get_path(cls, name):
    return cls.mapping['paths'].get(name, None)
