def __init__(self) ->None:
    self.model_zoo = {k: list(v.PRETRAINED_MODEL_CONFIG_DICT.keys()) for k,
        v in registry.mapping['model_name_mapping'].items()}
