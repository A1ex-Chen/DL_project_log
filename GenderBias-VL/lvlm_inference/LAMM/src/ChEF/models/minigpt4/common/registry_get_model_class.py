@classmethod
def get_model_class(cls, name):
    return cls.mapping['model_name_mapping'].get(name, None)
