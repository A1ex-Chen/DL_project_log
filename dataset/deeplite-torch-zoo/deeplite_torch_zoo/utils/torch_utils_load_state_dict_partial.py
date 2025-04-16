def load_state_dict_partial(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in
        model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    LOGGER.info(f'Loaded {len(pretrained_dict)}/{len(model_dict)} modules')
