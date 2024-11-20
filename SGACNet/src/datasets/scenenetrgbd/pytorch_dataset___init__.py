def __init__(self, data_dir=None, n_classes=13, split='train', depth_mode=
    'refined', with_input_orig=False):
    super(SceneNetRGBD, self).__init__()
    assert split in self.SPLITS
    assert n_classes == self.N_CLASSES
    assert depth_mode == 'refined'
    self._n_classes = n_classes
    self._split = split
    self._depth_mode = depth_mode
    self._with_input_orig = with_input_orig
    self._cameras = ['camera1']
    if data_dir is not None:
        data_dir = os.path.expanduser(data_dir)
        assert os.path.exists(data_dir)
        self._data_dir = data_dir

        def _loadtxt(fn):
            return np.loadtxt(os.path.join(self._data_dir, fn), dtype=str)
        self._files = {'rgb': _loadtxt(f'{self._split}_rgb.txt'), 'depth':
            _loadtxt(f'{self._split}_depth.txt'), 'label': _loadtxt(
            f'{self._split}_labels_{self._n_classes}.txt')}
        assert all(len(l) == len(self._files['rgb']) for l in self._files.
            values())
    else:
        print(f'Loaded {self.__class__.__name__} dataset without files')
    self._class_names = self.CLASS_NAMES
    self._class_colors = np.array(self.CLASS_COLORS, dtype='uint8')
    self._depth_mean = 4006.9281155769777
    self._depth_std = 2459.7763971709933
