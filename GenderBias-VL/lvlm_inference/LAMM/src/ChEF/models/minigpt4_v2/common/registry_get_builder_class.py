@classmethod
def get_builder_class(cls, name):
    return cls.mapping['builder_name_mapping'].get(name, None)
