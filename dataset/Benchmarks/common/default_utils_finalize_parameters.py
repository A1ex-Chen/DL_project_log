def finalize_parameters(bmk):
    """Utility to parse parameters in common as well as parameters
    particular to each benchmark.

    Parameters
    ----------
    bmk : benchmark object
        Object that has benchmark filepaths and specifications

    Return
    ----------
    gParameters : python dictionary
        Dictionary with all the parameters necessary to run the benchmark.
        Command line overwrites config file specifications
    """
    bmk.parse_from_common()
    bmk.parse_from_benchmark()
    aux = bmk.parser.parse_known_args()
    try:
        conffile_txt = aux[0].config_file
    except AttributeError:
        conffile = bmk.conffile
    else:
        conffile = os.path.join(bmk.file_path, conffile_txt)
    fileParameters = bmk.read_config_file(conffile)
    args = bmk.parser.parse_args()
    bmk_dict = bmk.additional_definitions
    check_file_parameters_exists(args, bmk_dict, fileParameters)
    gParameters = args_overwrite_config(args, fileParameters)
    bmk.check_required_exists(gParameters)
    print('Params:')
    pprint(gParameters)
    check_flag_conflicts(gParameters)
    return gParameters
