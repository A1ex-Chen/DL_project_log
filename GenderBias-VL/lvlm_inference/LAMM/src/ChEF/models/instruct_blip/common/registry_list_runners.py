@classmethod
def list_runners(cls):
    return sorted(cls.mapping['runner_name_mapping'].keys())
