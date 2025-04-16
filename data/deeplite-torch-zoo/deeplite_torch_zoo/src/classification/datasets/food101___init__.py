def __init__(self, root: str, split: str='train', transform=None,
    target_transform=None, download: bool=False) ->None:
    super().__init__(root, transform=transform, target_transform=
        target_transform)
    self._split = verify_str_arg(split, 'split', ('train', 'test'))
    self._base_folder = Path(self.root) / 'food-101'
    self._meta_folder = self._base_folder / 'meta'
    self._images_folder = self._base_folder / 'images'
    if download:
        self._download()
    if not self._check_exists():
        raise RuntimeError(
            'Dataset not found. You can use download=True to download it')
    self._labels = []
    self._image_files = []
    with open(self._meta_folder / f'{split}.json', encoding='utf8') as f:
        metadata = json.loads(f.read())
    self.classes = sorted(metadata.keys())
    self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
    for class_label, im_rel_paths in metadata.items():
        self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
        self._image_files += [self._images_folder.joinpath(*
            f'{im_rel_path}.jpg'.split('/')) for im_rel_path in im_rel_paths]
