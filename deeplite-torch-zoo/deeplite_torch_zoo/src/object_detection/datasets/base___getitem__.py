def __getitem__(self, index):
    """Returns transformed label information for given index."""
    return self.transforms(self.get_image_and_label(index))
