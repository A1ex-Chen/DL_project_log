def __init__(self, data_dir=None, split='train', depth_mode='refined',
    with_input_orig=False):
    super(SUNRGBD, self).__init__()
    self._n_classes = self.N_CLASSES
    self._cameras = ['realsense', 'kv2', 'kv1', 'xtion']
    assert split in self.SPLITS, f'parameter split must be one of {self.SPLITS}, got {split}'
    self._split = split
    assert depth_mode in ['refined', 'raw']
    self._depth_mode = depth_mode
    self._with_input_orig = with_input_orig
    if data_dir is not None:
        data_dir = os.path.expanduser(data_dir)
        self._data_dir = data_dir
        self.img_dir, self.depth_dir, self.label_dir = self.load_file_lists()
    else:
        print(f'Loaded {self.__class__.__name__} dataset without files')
    self._class_names = self.CLASS_NAMES_ENGLISH
    self._class_colors = np.array(self.CLASS_COLORS, dtype='uint8')
    self._depth_mean = 19025.14930492213
    self._depth_std = 9880.916071806689
