def reorg_datasets_by_split(datasets, batch_sizes):
    """
    Organizes datasets by split.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by name.

    Returns:
        Dict of datasets by split {split_name: List[Datasets]}.
    """
    reorg_datasets = dict()
    reorg_batch_sizes = dict()
    for dataset_name, dataset in datasets.items():
        for split_name, dataset_split in dataset.items():
            if split_name not in reorg_datasets:
                reorg_datasets[split_name] = [dataset_split]
                reorg_batch_sizes[split_name] = [batch_sizes[dataset_name]]
            else:
                reorg_datasets[split_name].append(dataset_split)
                reorg_batch_sizes[split_name].append(batch_sizes[dataset_name])
    return reorg_datasets, reorg_batch_sizes
