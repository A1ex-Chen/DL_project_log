def get_machine_id_list_for_test(target_dir, dir_name='test', ext='wav'):
    """
    target_dir : str
        base directory path of "dev_data" or "eval_data"
    test_dir_name : str (default="test")
        directory containing test data
    ext : str (default="wav)
        file extension of audio files

    return :
        machine_id_list : list [ str ]
            list of machine IDs extracted from the names of test files
    """
    dir_path = os.path.abspath('{dir}/{dir_name}/*.{ext}'.format(dir=
        target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(dir_path))
    machine_id_list = sorted(list(set(itertools.chain.from_iterable([re.
        findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list
