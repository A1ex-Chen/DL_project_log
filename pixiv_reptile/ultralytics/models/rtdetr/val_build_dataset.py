def build_dataset(self, img_path, mode='val', batch=None):
    """
        Build an RTDETR Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
    return RTDETRDataset(img_path=img_path, imgsz=self.args.imgsz,
        batch_size=batch, augment=False, hyp=self.args, rect=False, cache=
        self.args.cache or None, prefix=colorstr(f'{mode}: '), data=self.data)
