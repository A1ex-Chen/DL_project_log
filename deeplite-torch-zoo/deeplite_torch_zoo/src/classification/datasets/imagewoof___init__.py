def __init__(self, root: str, split: str='train', transform=None,
    target_transform=None, download: bool=False, url: str='',
    map_to_imagenet_labels=False) ->None:
    super().__init__(root, transform=transform, target_transform=
        target_transform)
    self._split = verify_str_arg(split, 'split', ('train', 'val'))
    self.url = url
    self._base_folder = Path(self.root) / url.split('/')[-1].replace('.zip', ''
        )
    if download:
        self._download()
    if not self._check_exists():
        raise RuntimeError(
            'Dataset not found. You can use download=True to download it')
    self._labels = []
    self._image_files = []
    self.classes = sorted(entry.name for entry in os.scandir(self.
        _base_folder / 'train') if entry.is_dir())
    self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)
        }
    for target_class in sorted(self.class_to_idx.keys()):
        class_index = self.class_to_idx[target_class]
        target_dir = os.path.join(self._base_folder / f'{split}', target_class)
        for folder, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(folder, fname)
                self._labels.append(class_index)
                self._image_files.append(path)
    self._map_to_imagenet_labels = map_to_imagenet_labels
