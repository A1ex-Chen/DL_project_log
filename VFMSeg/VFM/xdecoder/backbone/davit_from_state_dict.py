def from_state_dict(self, pretrained_dict, pretrained_layers=[], verbose=True):
    model_dict = self.state_dict()
    stripped_key = lambda x: x[14:] if x.startswith('image_encoder.') else x
    full_key_mappings = self._try_remap_keys(pretrained_dict)
    pretrained_dict = {stripped_key(full_key_mappings[k]): v for k, v in
        pretrained_dict.items() if stripped_key(full_key_mappings[k]) in
        model_dict.keys()}
    need_init_state_dict = {}
    for k, v in pretrained_dict.items():
        need_init = k.split('.')[0] in pretrained_layers or pretrained_layers[0
            ] == '*'
        if need_init:
            if verbose:
                print(f'=> init {k} from pretrained state dict')
            need_init_state_dict[k] = v
    self.load_state_dict(need_init_state_dict, strict=False)
