def __init__(self, data_dir, index_file_dir, split='train', num_classes=
    None, image_size=224, num_channels=3, batch_size=128, dtype='float32',
    one_hot=False, use_dali=False, augmenter=None, shuffle_buffer_size=
    10000, file_shuffle_buffer_size=1024, cache=False, mean_subtract=False,
    standardize=False, augmenter_params=None, mixup_alpha=0.0):
    """Initialize the builder from the config."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError('Cannot find data dir: {}'.format(data_dir))
    if one_hot and num_classes is None:
        raise FileNotFoundError('Number of classes is required for one_hot')
    self._data_dir = data_dir
    self._split = split
    self._image_size = image_size
    self._num_classes = num_classes
    self._num_channels = num_channels
    self._batch_size = batch_size
    self._dtype = dtype
    self._one_hot = one_hot
    self._augmenter_name = augmenter
    self._shuffle_buffer_size = shuffle_buffer_size
    self._file_shuffle_buffer_size = file_shuffle_buffer_size
    self._cache = cache
    self._mean_subtract = mean_subtract
    self._standardize = standardize
    self._index_file = index_file_dir
    self._use_dali = use_dali
    self.mixup_alpha = mixup_alpha
    self._num_gpus = sdp.size()
    if self._augmenter_name is not None:
        augmenter = AUGMENTERS.get(self._augmenter_name, None)
        params = augmenter_params or {}
        self._augmenter = augmenter(**params
            ) if augmenter is not None else None
    else:
        self._augmenter = None
