def load_tf_weights_in_openai_gpt(model, config, openai_checkpoint_folder_path
    ):
    """ Load tf pre-trained weights in a pytorch model (from NumPy arrays here)
    """
    import re
    import numpy as np
    if '.ckpt' in openai_checkpoint_folder_path:
        openai_checkpoint_folder_path = os.path.dirname(
            openai_checkpoint_folder_path)
    logger.info('Loading weights from {}'.format(openai_checkpoint_folder_path)
        )
    names = json.load(open(openai_checkpoint_folder_path +
        '/parameters_names.json', 'r', encoding='utf-8'))
    shapes = json.load(open(openai_checkpoint_folder_path +
        '/params_shapes.json', 'r', encoding='utf-8'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(openai_checkpoint_folder_path + '/params_{}.npy'
        .format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params,
        shapes)]
    init_params = [arr.squeeze() for arr in init_params]
    try:
        assert model.tokens_embed.weight.shape == init_params[1].shape
        assert model.positions_embed.weight.shape == init_params[0].shape
    except AssertionError as e:
        e.args += model.tokens_embed.weight.shape, init_params[1].shape
        e.args += model.positions_embed.weight.shape, init_params[0].shape
        raise
    model.tokens_embed.weight.data = torch.from_numpy(init_params[1])
    model.positions_embed.weight.data = torch.from_numpy(init_params[0])
    names.pop(0)
    init_params.pop(0)
    init_params.pop(0)
    for name, array in zip(names, init_params):
        name = name[6:]
        assert name[-2:] == ':0'
        name = name[:-2]
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+\\d+', m_name):
                l = re.split('(\\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'g':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'b':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'w':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += pointer.shape, array.shape
            raise
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += pointer.shape, array.shape
            raise
        logger.info('Initialize PyTorch weight {}'.format(name))
        pointer.data = torch.from_numpy(array)
    return model
