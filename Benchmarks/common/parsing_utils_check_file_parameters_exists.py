def check_file_parameters_exists(params_parser, params_benchmark, params_file):
    """Functionality to verify that the parameters defined in the configuraion file are recognizable by the command line parser (i.e. no uknown keywords are used in the configuration file).

    Parameters
    ----------
    params_parser : python dictionary
        Includes parameters set via the command line.
    params_benchmark : python list
        Includes additional parameters defined in the benchmark.
    params_file : python dictionary
        Includes parameters read from the configuration file.

        Global:
        PARAMETERS_CANDLE : python list
            Includes all the core keywords that are specified in CANDLE.
    """
    args_dict = vars(params_parser)
    args_set = set(args_dict.keys())
    bmk_keys = []
    for item in params_benchmark:
        bmk_keys.append(item['name'])
    bmk_set = set(bmk_keys)
    candle_set = set(PARAMETERS_CANDLE)
    candle_set = candle_set.union(args_set)
    candle_set = candle_set.union(bmk_set)
    file_set = set(params_file.keys())
    diff_set = file_set.difference(candle_set)
    if len(diff_set) > 0:
        message = (
            'These keywords used in the configuration file are not defined in CANDLE: '
             + str(sorted(diff_set)))
        warnings.warn(message, RuntimeWarning)
