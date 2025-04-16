def __init__(self, root: str, split: str='train', transform=None,
    target_transform=None, download: bool=False) ->None:
    super().__init__(root, transform=transform, target_transform=
        target_transform)
    self._split = verify_str_arg(split, 'split', ('train', 'val', 'test'))
    self._base_folder = Path(self.root) / 'flowers-102'
    self._images_folder = self._base_folder / 'jpg'
    if download:
        self.download()
    if not self._check_integrity():
        raise RuntimeError(
            'Dataset not found or corrupted. You can use download=True to download it'
            )
    from scipy.io import loadmat
    set_ids = loadmat(self._base_folder / self._file_dict['setid'][0],
        squeeze_me=True)
    image_ids = set_ids[self._splits_map[self._split]].tolist()
    labels = loadmat(self._base_folder / self._file_dict['label'][0],
        squeeze_me=True)
    image_id_to_label = dict(enumerate((labels['labels'] - 1).tolist(), 1))
    self._labels = []
    self._image_files = []
    for image_id in image_ids:
        self._labels.append(image_id_to_label[image_id])
        self._image_files.append(self._images_folder /
            f'image_{image_id:05d}.jpg')
    self.classes = set(self._labels)
