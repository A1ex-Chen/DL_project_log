def build_dataset(self, img_path, mode='val', batch=None):
    """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
    return build_yolo_dataset(self.args, img_path, batch, self.data, mode=
        mode, stride=self.stride)
