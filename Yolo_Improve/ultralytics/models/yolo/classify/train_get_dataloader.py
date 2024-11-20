def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
    """Returns PyTorch DataLoader with transforms to preprocess images for inference."""
    with torch_distributed_zero_first(rank):
        dataset = self.build_dataset(dataset_path, mode)
    loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank
        )
    if mode != 'train':
        if is_parallel(self.model):
            self.model.module.transforms = loader.dataset.torch_transforms
        else:
            self.model.transforms = loader.dataset.torch_transforms
    return loader
