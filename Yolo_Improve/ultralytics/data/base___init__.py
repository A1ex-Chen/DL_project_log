def __init__(self, img_path, imgsz=640, cache=False, augment=True, hyp=
    DEFAULT_CFG, prefix='', rect=False, batch_size=16, stride=32, pad=0.5,
    single_cls=False, classes=None, fraction=1.0):
    """Initialize BaseDataset with given configuration and options."""
    super().__init__()
    self.img_path = img_path
    self.imgsz = imgsz
    self.augment = augment
    self.single_cls = single_cls
    self.prefix = prefix
    self.fraction = fraction
    self.im_files = self.get_img_files(self.img_path)
    self.labels = self.get_labels()
    self.update_labels(include_class=classes)
    self.ni = len(self.labels)
    self.rect = rect
    self.batch_size = batch_size
    self.stride = stride
    self.pad = pad
    if self.rect:
        assert self.batch_size is not None
        self.set_rectangle()
    self.buffer = []
    self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)
        ) if self.augment else 0
    self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [
        None] * self.ni
    self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
    self.cache = cache.lower() if isinstance(cache, str
        ) else 'ram' if cache is True else None
    if self.cache == 'ram' and self.check_cache_ram() or self.cache == 'disk':
        self.cache_images()
    self.transforms = self.build_transforms(hyp=hyp)