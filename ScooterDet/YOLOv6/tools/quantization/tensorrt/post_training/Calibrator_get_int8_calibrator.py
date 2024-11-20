def get_int8_calibrator(calib_cache, calib_data, max_calib_size,
    calib_batch_size):
    if os.path.exists(calib_cache):
        logger.info('Skipping calibration files, using calibration cache: {:}'
            .format(calib_cache))
        calib_files = []
    else:
        if not calib_data:
            raise ValueError(
                'ERROR: Int8 mode requested, but no calibration data provided. Please provide --calibration-data /path/to/calibration/files'
                )
        calib_files = get_calibration_files(calib_data, max_calib_size)
    preprocess_func = preprocess_yolov6
    int8_calibrator = ImageCalibrator(calibration_files=calib_files,
        batch_size=calib_batch_size, cache_file=calib_cache)
    return int8_calibrator
