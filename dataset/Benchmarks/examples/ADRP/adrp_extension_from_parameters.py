def extension_from_parameters(params, framework=''):
    """Construct string for saving model with annotation of parameters"""
    ext = framework
    for i, n in enumerate(params['dense']):
        if n:
            ext += '.D{}={}'.format(i + 1, n)
    ext += '.A={}'.format(params['activation'])
    ext += '.OA={}'.format(params['out_activation'])
    ext += '.B={}'.format(params['batch_size'])
    ext += '.E={}'.format(params['epochs'])
    ext += '.L={}'.format(params['latent_dim'])
    ext += '.LR={}'.format(params['learning_rate'])
    ext += '.S={}'.format(params['scaling'])
    if params['epsilon_std'] != 1.0:
        ext += '.EPS={}'.format(params['epsilon_std'])
    if params['dropout']:
        ext += '.DR={}'.format(params['dropout'])
    if params['batch_normalization']:
        ext += '.BN'
    if params['warmup_lr']:
        ext += '.WU_LR'
    if params['reduce_lr']:
        ext += '.Re_LR'
    if params['residual']:
        ext += '.Res'
    return ext
