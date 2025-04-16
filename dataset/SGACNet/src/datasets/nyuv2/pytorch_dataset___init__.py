def __init__(self, data_dir=None, n_classes=40, split='train', depth_mode=
    'refined', with_input_orig=False):
    super(NYUv2, self).__init__()
    assert split in self.SPLITS
    assert n_classes in self.N_CLASSES
    assert depth_mode in ['refined', 'raw']
    self._n_classes = n_classes
    self._split = split
    self._depth_mode = depth_mode
    self._with_input_orig = with_input_orig
    self._cameras = ['kv1']
    if data_dir is not None:
        data_dir = os.path.expanduser(data_dir)
        assert os.path.exists(data_dir)
        self._data_dir = data_dir
        fp = os.path.join(self._data_dir, self.SPLIT_FILELIST_FILENAMES[
            self._split])
        self._filenames = np.loadtxt(fp, dtype=str)
    else:
        print(f'Loaded {self.__class__.__name__} dataset without files')
    self._class_names = getattr(self, f'CLASS_NAMES_{self._n_classes}')
    self._class_colors = np.array(getattr(self,
        f'CLASS_COLORS_{self._n_classes}'), dtype='uint8')
    self._depth_mean = 2841.94941272766
    self._depth_std = 1417.2594281672277
