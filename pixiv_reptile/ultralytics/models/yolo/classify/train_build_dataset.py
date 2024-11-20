def build_dataset(self, img_path, mode='train', batch=None):
    """Creates a ClassificationDataset instance given an image path, and mode (train/create_self_data etc.)."""
    return ClassificationDataset(root=img_path, args=self.args, augment=
        mode == 'train', prefix=mode)
