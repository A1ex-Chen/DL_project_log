def test_file_list_generator(target_dir, id_name, mode, dir_name='test',
    prefix_normal='normal', prefix_anomaly='anomaly', ext='wav'):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    id_name : str
        id of wav file in <<test_dir_name>> directory
    dir_name : str (default="test")
        directory containing test data
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files

    return :
        if the mode is "development":
            test_files : list [ str ]
                file list for test
            test_labels : list [ boolean ]
                label info. list for test
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            test_files : list [ str ]
                file list for test
    """
    logger.info('target_dir : {}'.format(target_dir + '_' + id_name))
    if mode:
        normal_files = sorted(glob.glob(
            '{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}'.format(dir=
            target_dir, dir_name=dir_name, prefix_normal=prefix_normal,
            id_name=id_name, ext=ext)))
        normal_labels = numpy.zeros(len(normal_files))
        anomaly_files = sorted(glob.glob(
            '{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}'.format(dir
            =target_dir, dir_name=dir_name, prefix_anomaly=prefix_anomaly,
            id_name=id_name, ext=ext)))
        anomaly_labels = numpy.ones(len(anomaly_files))
        files = numpy.concatenate((normal_files, anomaly_files), axis=0)
        labels = numpy.concatenate((normal_labels, anomaly_labels), axis=0)
        logger.info('test_file  num : {num}'.format(num=len(files)))
        if len(files) == 0:
            logger.exception('no_wav_file!!')
        print('\n========================================')
    else:
        files = sorted(glob.glob('{dir}/{dir_name}/*{id_name}*.{ext}'.
            format(dir=target_dir, dir_name=dir_name, id_name=id_name, ext=
            ext)))
        labels = None
        logger.info('test_file  num : {num}'.format(num=len(files)))
        if len(files) == 0:
            logger.exception('no_wav_file!!')
        print('\n=========================================')
    return files, labels
