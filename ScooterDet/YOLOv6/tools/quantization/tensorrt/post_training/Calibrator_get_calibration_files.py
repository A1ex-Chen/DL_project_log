def get_calibration_files(calibration_data, max_calibration_size=None,
    allowed_extensions=('.jpeg', '.jpg', '.png')):
    """Returns a list of all filenames ending with `allowed_extensions` found in the `calibration_data` directory.

    Parameters
    ----------
    calibration_data: str
        Path to directory containing desired files.
    max_calibration_size: int
        Max number of files to use for calibration. If calibration_data contains more than this number,
        a random sample of size max_calibration_size will be returned instead. If None, all samples will be used.

    Returns
    -------
    calibration_files: List[str]
         List of filenames contained in the `calibration_data` directory ending with `allowed_extensions`.
    """
    logger.info('Collecting calibration files from: {:}'.format(
        calibration_data))
    calibration_files = [path for path in glob.iglob(os.path.join(
        calibration_data, '**'), recursive=True) if os.path.isfile(path) and
        path.lower().endswith(allowed_extensions)]
    logger.info('Number of Calibration Files found: {:}'.format(len(
        calibration_files)))
    if len(calibration_files) == 0:
        raise Exception('ERROR: Calibration data path [{:}] contains no files!'
            .format(calibration_data))
    if max_calibration_size:
        if len(calibration_files) > max_calibration_size:
            logger.warning(
                'Capping number of calibration images to max_calibration_size: {:}'
                .format(max_calibration_size))
            random.seed(42)
            calibration_files = random.sample(calibration_files,
                max_calibration_size)
    return calibration_files
