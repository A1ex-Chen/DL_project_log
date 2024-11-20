@classmethod
def list_datasets(cls):
    return sorted(cls.mapping['builder_name_mapping'].keys())
