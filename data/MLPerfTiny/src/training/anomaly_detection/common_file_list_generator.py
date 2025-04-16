def file_list_generator(target_dir, dir_name='train', ext='wav'):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files

    return :
        train_files : list [ str ]
            file list for training
    """
    logger.info('target_dir : {}'.format(target_dir))
    training_list_path = os.path.abspath('{dir}/{dir_name}/*.{ext}'.format(
        dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        logger.exception('no_wav_file!!')
    logger.info('train_file num : {num}'.format(num=len(files)))
    return files
