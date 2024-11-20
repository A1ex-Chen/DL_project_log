def load_file(filename, mmap_mode=None, verbose=True, allow_pickle=False):
    """
    Common i/o utility to handle loading data from various file formats.
    Supported:
        .pkl, .pickle, .npy, .json
    For the npy files, we support reading the files in mmap_mode.
    If the mmap_mode of reading is not successful, we load data without the
    mmap_mode.
    """
    if verbose:
        logging.info(f'Loading data from file: {filename}')
    file_ext = os.path.splitext(filename)[1]
    if file_ext == '.txt':
        with g_pathmgr.open(filename, 'r') as fopen:
            data = fopen.readlines()
    elif file_ext in ['.pkl', '.pickle']:
        with g_pathmgr.open(filename, 'rb') as fopen:
            data = pickle.load(fopen, encoding='latin1')
    elif file_ext == '.npy':
        if mmap_mode:
            try:
                with g_pathmgr.open(filename, 'rb') as fopen:
                    data = np.load(fopen, allow_pickle=allow_pickle,
                        encoding='latin1', mmap_mode=mmap_mode)
            except ValueError as e:
                logging.info(
                    f'Could not mmap {filename}: {e}. Trying without g_pathmgr'
                    )
                data = np.load(filename, allow_pickle=allow_pickle,
                    encoding='latin1', mmap_mode=mmap_mode)
                logging.info('Successfully loaded without g_pathmgr')
            except Exception:
                logging.info(
                    'Could not mmap without g_pathmgr. Trying without mmap')
                with g_pathmgr.open(filename, 'rb') as fopen:
                    data = np.load(fopen, allow_pickle=allow_pickle,
                        encoding='latin1')
        else:
            with g_pathmgr.open(filename, 'rb') as fopen:
                data = np.load(fopen, allow_pickle=allow_pickle, encoding=
                    'latin1')
    elif file_ext == '.json':
        with g_pathmgr.open(filename, 'r') as fopen:
            data = json.load(fopen)
    elif file_ext == '.yaml':
        with g_pathmgr.open(filename, 'r') as fopen:
            data = yaml.load(fopen, Loader=yaml.FullLoader)
    elif file_ext == '.csv':
        with g_pathmgr.open(filename, 'r') as fopen:
            data = pd.read_csv(fopen)
    else:
        raise Exception(f'Reading from {file_ext} is not supported yet')
    return data
