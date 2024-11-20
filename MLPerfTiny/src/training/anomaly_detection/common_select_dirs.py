def select_dirs(param, mode):
    """
    param : dict
        baseline.yaml data

    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        logger.info('load_directory <- development')
        dir_path = os.path.abspath('{base}/*'.format(base=param[
            'dev_directory']))
        dirs = sorted(glob.glob(dir_path))
    else:
        logger.info('load_directory <- evaluation')
        dir_path = os.path.abspath('{base}/*'.format(base=param[
            'eval_directory']))
        dirs = sorted(glob.glob(dir_path))
    return dirs
