def sample_dataset(dataset, sample_len=1000, sample_seed=0, dist_args=None):
    if sample_len == -1:
        pass
    elif len(dataset) > sample_len:
        np.random.seed(sample_seed)
        random_indices = np.random.choice(len(dataset), sample_len, replace
            =False)
        dataset = CustomSubset(dataset, random_indices)
    if dist_args is not None and dist_args['world_size'] > 1:
        rank = dist_args['global_rank']
        world_size = dist_args['world_size']
        data_per_rank = len(dataset) // world_size
        start = rank * data_per_rank
        end = (rank + 1) * data_per_rank
        if rank == world_size - 1:
            end = len(dataset)
        if isinstance(dataset, CustomSubset):
            original_indices = dataset.indices
            sliced_indices = original_indices[start:end]
            dataset = CustomSubset(dataset.dataset, sliced_indices)
        else:
            sliced_indices = [i for i in range(len(dataset))][start:end]
            dataset = CustomSubset(dataset, sliced_indices)
    return dataset
