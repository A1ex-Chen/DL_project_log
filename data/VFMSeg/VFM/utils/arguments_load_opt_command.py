def load_opt_command(args):
    parser = argparse.ArgumentParser(description=
        'Pretrain or fine-tune models for NLP tasks.')
    parser.add_argument('command', help=
        'Command: train/evaluate/train-and-evaluate')
    parser.add_argument('--conf_files', required=True, help=
        'Path(s) to the config file(s).')
    parser.add_argument('--config_overrides', nargs='*', help=
        'Override parameters on config with a json style string, e.g. {"<PARAM_NAME_1>": <PARAM_VALUE_1>, "<PARAM_GROUP_2>.<PARAM_SUBGROUP_2>.<PARAM_2>": <PARAM_VALUE_2>}. A key with "." updates the object in the corresponding nested dict. Remember to escape " in command line.'
        )
    parser.add_argument('--overrides', help=
        'arguments that used to overide the config file in cmdline', nargs=
        argparse.REMAINDER)
    cmdline_args = parser.parse_args() if not args else parser.parse_args(args)
    opt = load_opt_from_config_files(cmdline_args.conf_files)
    if cmdline_args.config_overrides:
        config_overrides_string = ' '.join(cmdline_args.config_overrides)
        logger.warning(
            f'Command line config overrides: {config_overrides_string}')
        config_dict = json.loads(config_overrides_string)
        load_config_dict_to_opt(opt, config_dict)
    if cmdline_args.overrides:
        assert len(cmdline_args.overrides
            ) % 2 == 0, 'overides arguments is not paired, required: key value'
        keys = [cmdline_args.overrides[idx * 2] for idx in range(len(
            cmdline_args.overrides) // 2)]
        vals = [cmdline_args.overrides[idx * 2 + 1] for idx in range(len(
            cmdline_args.overrides) // 2)]
        vals = [(val.replace('false', '').replace('False', '') if len(val.
            replace(' ', '')) == 5 else val) for val in vals]
        types = []
        for key in keys:
            key = key.split('.')
            ele = opt.copy()
            while len(key) > 0:
                ele = ele[key.pop(0)]
            types.append(type(ele))
        config_dict = {x: z(y) for x, y, z in zip(keys, vals, types)}
        load_config_dict_to_opt(opt, config_dict)
    for key, val in cmdline_args.__dict__.items():
        if val is not None:
            opt[key] = val
    return opt, cmdline_args
