def config_env(args):
    args['root_dir'] = '../'
    args['mode'] = 'train'
    initialize_distributed(args)
    set_random_seed(args['seed'])
