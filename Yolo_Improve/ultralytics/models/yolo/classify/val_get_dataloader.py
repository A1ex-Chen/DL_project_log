def get_dataloader(self, dataset_path, batch_size):
    """Builds and returns a data loader for classification tasks with given parameters."""
    dataset = self.build_dataset(dataset_path)
    return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)
