def get_model_test_files():
    """Get the model test files.

    The returned files should NOT contain the `tests` (i.e. `PATH_TO_TESTS` defined in this script). They will be
    considered as paths relative to `tests`. A caller has to use `os.path.join(PATH_TO_TESTS, ...)` to access the files.
    """
    _ignore_files = ['test_modeling_common',
        'test_modeling_encoder_decoder',
        'test_modeling_flax_encoder_decoder',
        'test_modeling_flax_speech_encoder_decoder', 'test_modeling_marian',
        'test_modeling_tf_common', 'test_modeling_tf_encoder_decoder']
    test_files = []
    model_test_root = os.path.join(PATH_TO_TESTS, 'models')
    model_test_dirs = []
    for x in os.listdir(model_test_root):
        x = os.path.join(model_test_root, x)
        if os.path.isdir(x):
            model_test_dirs.append(x)
    for target_dir in ([PATH_TO_TESTS] + model_test_dirs):
        for file_or_dir in os.listdir(target_dir):
            path = os.path.join(target_dir, file_or_dir)
            if os.path.isfile(path):
                filename = os.path.split(path)[-1]
                if 'test_modeling' in filename and os.path.splitext(filename)[0
                    ] not in _ignore_files:
                    file = os.path.join(*path.split(os.sep)[1:])
                    test_files.append(file)
    return test_files
