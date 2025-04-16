def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
    """Construct and return dataloader."""
    assert mode in {'train', 'val'
        }, f"Mode must be 'train' or 'val', not {mode}."
    with torch_distributed_zero_first(rank):
        dataset = self.build_dataset(dataset_path, mode, batch_size)
    shuffle = mode == 'train'
    if getattr(dataset, 'rect', False) and shuffle:
        LOGGER.warning(
            "WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False"
            )
        shuffle = False
    workers = self.args.workers if mode == 'train' else self.args.workers * 2
    return build_dataloader(dataset, batch_size, workers, shuffle, rank)
