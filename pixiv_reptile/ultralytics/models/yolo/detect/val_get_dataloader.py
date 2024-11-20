def get_dataloader(self, dataset_path, batch_size):
    """Construct and return dataloader."""
    dataset = self.build_dataset(dataset_path, batch=batch_size, mode='val')
    return build_dataloader(dataset, batch_size, self.args.workers, shuffle
        =False, rank=-1)
