def __init__(self, dataset: Union[Dataset, ConcatDataset], batch_size: int,
    num_workers: int):
    super().__init__()
    self._dataset = dataset
    self._dataloader = DataLoader(dataset, batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=Dataset.collate_fn, pin_memory=True
        )
    if isinstance(self._dataset, Dataset):
        self._num_classes = self._dataset.num_classes()
    elif isinstance(self._dataset, ConcatDataset):
        self._num_classes = self._dataset.master.num_classes()
    else:
        raise TypeError
