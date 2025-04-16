def clr_set_args(args):
    req_keys = ['clr_mode', 'clr_base_lr', 'clr_max_lr', 'clr_gamma']
    exclusive_keys = ['warmup_lr', 'reduce_lr']
    keys_present = True
    for key in req_keys:
        if key not in args.keys():
            keys_present = False
    if keys_present and args['clr_mode'] is not None:
        clr_keras_kwargs = {'mode': args['clr_mode'], 'base_lr': args[
            'clr_base_lr'], 'max_lr': args['clr_max_lr'], 'gamma': args[
            'clr_gamma']}
        for ex_key in exclusive_keys:
            if ex_key in args.keys():
                if args[ex_key] is True:
                    print('Key ', ex_key, ' conflicts, setting to False')
                    args[ex_key] = False
    else:
        print('Incomplete CLR specification: will run without')
        clr_keras_kwargs = {'mode': None, 'base_lr': 0.1, 'max_lr': 0.1,
            'gamma': 0.1}
    return clr_keras_kwargs
