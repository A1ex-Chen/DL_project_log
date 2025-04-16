@classmethod
def get_processor_class(cls, name):
    return cls.mapping['processor_name_mapping'].get(name, None)
