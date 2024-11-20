@staticmethod
def collate_fn(batch):
    """Collates data samples into batches."""
    return YOLODataset.collate_fn(batch)
