@classmethod
def list_processors(cls):
    return sorted(cls.mapping['processor_name_mapping'].keys())
