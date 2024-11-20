def __init__(self, root, split):
    self._check_split(split)
    self._check_download(root)
    self._load_data(split)
    self._load_vocab()
