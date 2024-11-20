def concat_datasets(datasets):
    """
    Concatenates multiple datasets into a single dataset.

    It supports may-style datasets and DataPipeline from WebDataset. Currently, does not support
    generic IterableDataset because it requires creating separate samplers.

    Now only supports conctenating training datasets and assuming validation and testing
    have only a single dataset. This is because metrics should not be computed on the concatenated
    datasets.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by split.

    Returns:
        Dict of concatenated datasets by split, "train" is the concatenation of multiple datasets,
        "val" and "test" remain the same.

        If the input training datasets contain both map-style and DataPipeline datasets, returns
        a tuple, where the first element is a concatenated map-style dataset and the second
        element is a chained DataPipeline dataset.

    """
    for split_name in datasets:
        if split_name != 'train':
            assert len(datasets[split_name]
                ) == 1, 'Do not support multiple {} datasets.'.format(
                split_name)
            datasets[split_name] = datasets[split_name][0]
        else:
            iterable_datasets, map_datasets = [], []
            for dataset in datasets[split_name]:
                if isinstance(dataset, wds.DataPipeline):
                    logging.info(
                        "Dataset {} is IterableDataset, can't be concatenated."
                        .format(dataset))
                    iterable_datasets.append(dataset)
                elif isinstance(dataset, IterableDataset):
                    raise NotImplementedError(
                        'Do not support concatenation of generic IterableDataset.'
                        )
                else:
                    map_datasets.append(dataset)
            if len(iterable_datasets) > 1:
                chained_datasets = ChainDataset(iterable_datasets)
            elif len(iterable_datasets) == 1:
                chained_datasets = iterable_datasets[0]
            else:
                chained_datasets = None
            concat_datasets = ConcatDataset(map_datasets) if len(map_datasets
                ) > 0 else None
            train_datasets = concat_datasets, chained_datasets
            train_datasets = tuple([x for x in train_datasets if x is not None]
                )
            train_datasets = train_datasets[0] if len(train_datasets
                ) == 1 else train_datasets
            datasets[split_name] = train_datasets
    return datasets
