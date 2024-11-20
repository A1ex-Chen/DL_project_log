def _make_train_val_dir(self):
    self._label_dir_path = os.path.join(self._save_path_pfx, 'labels/')
    self._image_dir_path = os.path.join(self._save_path_pfx, 'images/')
    for yolo_path in (os.path.join(self._label_dir_path + 'train/'), os.
        path.join(self._label_dir_path + 'val/'), os.path.join(self.
        _image_dir_path + 'train/'), os.path.join(self._image_dir_path +
        'val/')):
        if os.path.exists(yolo_path):
            shutil.rmtree(yolo_path)
        os.makedirs(yolo_path)
