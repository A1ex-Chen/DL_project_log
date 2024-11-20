def get_available_checkpoint_keys(model, ckpt):
    print('==> Attemp to reload from %s' % ckpt)
    state_dict = torch.load(ckpt)['state_dict']
    current_state_dict = model.state_dict()
    new_state_dict = {}
    for k in state_dict.keys():
        if k in current_state_dict.keys() and current_state_dict[k].size(
            ) == state_dict[k].size():
            new_state_dict[k] = state_dict[k]
        else:
            print('==> WARNING: Skipping %s' % k)
    print('%s out of %s keys are matched' % (len(new_state_dict.keys()),
        len(state_dict.keys())))
    return new_state_dict
