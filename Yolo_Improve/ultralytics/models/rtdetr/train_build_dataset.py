def build_dataset(self, img_path, mode='val', batch=None):
    """
        Build and return an RT-DETR dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): Dataset mode, either 'train' or 'val'.
            batch (int, optional): Batch size for rectangle training. Defaults to None.

        Returns:
            (RTDETRDataset): Dataset object for the specific mode.
        """
    return RTDETRDataset(img_path=img_path, imgsz=self.args.imgsz,
        batch_size=batch, augment=mode == 'train', hyp=self.args, rect=
        False, cache=self.args.cache or None, prefix=colorstr(f'{mode}: '),
        data=self.data)
