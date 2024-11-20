def restore(ckpt_path, model):
    if not Path(ckpt_path).is_file():
        raise ValueError('checkpoint {} not exist.'.format(ckpt_path))
    model.load_state_dict(torch.load(ckpt_path))
    print('Restoring parameters from {}'.format(ckpt_path))
