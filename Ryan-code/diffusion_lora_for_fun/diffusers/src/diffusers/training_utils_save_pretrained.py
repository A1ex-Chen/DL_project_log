def save_pretrained(self, path):
    if self.model_cls is None:
        raise ValueError(
            '`save_pretrained` can only be used if `model_cls` was defined at __init__.'
            )
    if self.model_config is None:
        raise ValueError(
            '`save_pretrained` can only be used if `model_config` was defined at __init__.'
            )
    model = self.model_cls.from_config(self.model_config)
    state_dict = self.state_dict()
    state_dict.pop('shadow_params', None)
    model.register_to_config(**state_dict)
    self.copy_to(model.parameters())
    model.save_pretrained(path)
