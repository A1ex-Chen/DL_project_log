@classmethod
def list_models(cls):
    return sorted(cls.mapping['model_name_mapping'].keys())
