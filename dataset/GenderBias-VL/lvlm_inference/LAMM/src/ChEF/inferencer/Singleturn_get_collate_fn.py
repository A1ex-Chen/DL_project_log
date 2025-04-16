def get_collate_fn(self, dataset):
    if hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    if hasattr(dataset, 'collate'):
        collate_fn = dataset.collate
    else:
        collate_fn = lambda batch: {key: [data[key] for data in batch] for
            key in batch[0]}
    return collate_fn
