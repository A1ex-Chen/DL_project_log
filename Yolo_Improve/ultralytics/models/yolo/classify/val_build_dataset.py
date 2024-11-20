def build_dataset(self, img_path):
    """Creates and returns a ClassificationDataset instance using given image path and preprocessing parameters."""
    return ClassificationDataset(root=img_path, args=self.args, augment=
        False, prefix=self.args.split)
