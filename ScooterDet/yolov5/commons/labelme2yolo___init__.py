def __init__(self, json_dir, to_seg=False):
    self._json_dir = json_dir
    self._label_id_map = {v: k for k, v in category_map.items()}
    self._to_seg = to_seg
    i = 'YOLODataset'
    i += '_seg/' if to_seg else '/'
    self._save_path_pfx = os.path.join(self._json_dir, i)
