def __init__(self, state_dict: Dict[str, torch.Tensor]):
    super().__init__()
    self.layers = torch.nn.ModuleList(state_dict.values())
    self.mapping = dict(enumerate(state_dict.keys()))
    self.rev_mapping = {v: k for k, v in enumerate(state_dict.keys())}
    self.split_keys = ['.processor', '.self_attn']

    def map_to(module, state_dict, *args, **kwargs):
        new_state_dict = {}
        for key, value in state_dict.items():
            num = int(key.split('.')[1])
            new_key = key.replace(f'layers.{num}', module.mapping[num])
            new_state_dict[new_key] = value
        return new_state_dict

    def remap_key(key, state_dict):
        for k in self.split_keys:
            if k in key:
                return key.split(k)[0] + k
        raise ValueError(
            f'There seems to be a problem with the state_dict: {set(state_dict.keys())}. {key} has to have one of {self.split_keys}.'
            )

    def map_from(module, state_dict, *args, **kwargs):
        all_keys = list(state_dict.keys())
        for key in all_keys:
            replace_key = remap_key(key, state_dict)
            new_key = key.replace(replace_key,
                f'layers.{module.rev_mapping[replace_key]}')
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    self._register_state_dict_hook(map_to)
    self._register_load_state_dict_pre_hook(map_from, with_module=True)
