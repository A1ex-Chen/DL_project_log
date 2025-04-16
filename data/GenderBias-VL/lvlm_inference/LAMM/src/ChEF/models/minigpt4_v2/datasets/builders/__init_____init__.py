def __init__(self) ->None:
    self.dataset_zoo = {k: list(v.DATASET_CONFIG_DICT.keys()) for k, v in
        sorted(registry.mapping['builder_name_mapping'].items())}
