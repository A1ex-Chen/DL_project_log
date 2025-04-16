def __init__(self, data_dir=None, n_classes=19, split='train', depth_mode=
    'raw', with_input_orig=False, disparity_instead_of_depth=True):
    super(Cityscapes, self).__init__()
    assert split in self.SPLITS
    assert n_classes in self.N_CLASSES
    assert depth_mode == 'raw'
    self._n_classes = n_classes
    self._split = split
    self._depth_mode = depth_mode
    self._with_input_orig = with_input_orig
    self._disparity_instead_of_depth = disparity_instead_of_depth
    self._cameras = ['camera1']
    if self._disparity_instead_of_depth:
        self._depth_dir = self.DISPARITY_RAW_DIR
    else:
        self._depth_dir = self.DEPTH_RAW_DIR
    if data_dir is not None:
        data_dir = os.path.expanduser(data_dir)
        assert os.path.exists(data_dir)
        self._data_dir = data_dir

        def _loadtxt(fn):
            return np.loadtxt(os.path.join(self._data_dir, fn), dtype=str)
        self._files = {'rgb': _loadtxt(f'{self._split}_rgb.txt'), self.
            _depth_dir: _loadtxt(f'{self._split}_{self._depth_dir}.txt'),
            'label': _loadtxt(f'{self._split}_labels_{self._n_classes}.txt')}
        assert all(len(l) == len(self._files['rgb']) for l in self._files.
            values())
    else:
        print(f'Loaded {self.__class__.__name__} dataset without files')
    if self._n_classes == 19:
        self._class_names = self.CLASS_NAMES_REDUCED
        self._class_colors = np.array(self.CLASS_COLORS_REDUCED, dtype='uint8')
        self._label_dir = self.LABELS_REDUCED_DIR
    else:
        self._class_names = self.CLASS_NAMES_FULL
        self._class_colors = np.array(self.CLASS_COLORS_FULL, dtype='uint8')
        self._label_dir = self.LABELS_FULL_DIR
    if disparity_instead_of_depth:
        self._depth_mean = 9069.706336834102
        self._depth_std = 7178.335960071306
    else:
        self._depth_mean = 31.715617493177906
        self._depth_std = 38.70280704877372
