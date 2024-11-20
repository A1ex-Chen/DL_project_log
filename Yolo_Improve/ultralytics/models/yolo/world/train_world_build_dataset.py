def build_dataset(self, img_path, mode='train', batch=None):
    """
        Build YOLO Dataset.

        Args:
            img_path (List[str] | str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
    gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32
        )
    if mode != 'train':
        return build_yolo_dataset(self.args, img_path, batch, self.data,
            mode=mode, rect=mode == 'val', stride=gs)
    dataset = [(build_yolo_dataset(self.args, im_path, batch, self.data,
        stride=gs, multi_modal=True) if isinstance(im_path, str) else
        build_grounding(self.args, im_path['img_path'], im_path['json_file'
        ], batch, stride=gs)) for im_path in img_path]
    return YOLOConcatDataset(dataset) if len(dataset) > 1 else dataset[0]
