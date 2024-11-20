def args_overwrite_config(args, config):
    """Overwrite configuration parameters with
    parameters specified via command-line.

    Parameters
    ----------
    args : ArgumentParser object
        Parameters specified via command-line
    config : python dictionary
        Parameters read from configuration file
    """
    params = config
    args_dict = vars(args)
    for key in args_dict.keys():
        params[key] = args_dict[key]
    if 'data_type' not in params:
        params['data_type'] = DEFAULT_DATATYPE
    elif params['data_type'] in set(['f16', 'f32', 'f64']):
        params['data_type'] = get_choice(params['datatype'])
    if 'output_dir' not in params:
        params['output_dir'] = directory_from_parameters(params)
    else:
        params['output_dir'] = directory_from_parameters(params, params[
            'output_dir'])
    if 'rng_seed' not in params:
        params['rng_seed'] = DEFAULT_SEED
    if 'timeout' not in params:
        params['timeout'] = DEFAULT_TIMEOUT
    return params
