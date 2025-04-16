def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
    """Returns dataloader derived from torch.data.Dataloader."""
    raise NotImplementedError(
        'get_dataloader function not implemented in trainer')
