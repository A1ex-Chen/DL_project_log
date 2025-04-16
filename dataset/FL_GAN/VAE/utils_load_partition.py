def load_partition(idx: int, my_dataset):
    """Load idx th of the training and test data to simulate a partition."""
    """idx means how many clients will join the learning, so that we split related subsets."""
    trainset, testset = load_data(my_dataset)
    n_train = int(len(trainset) / idx)
    n_test = int(len(testset) / idx)
    idx_start = random.randint(0, idx - 1)
    bound_train = (idx_start + 1) * n_train if (idx_start + 1) * n_train < len(
        trainset) else len(trainset)
    bound_test = (idx_start + 1) * n_test if (idx_start + 1) * n_test < len(
        testset) else len(testset)
    train_parition = torch.utils.data.Subset(trainset, range(idx_start *
        n_train, bound_train))
    test_parition = torch.utils.data.Subset(testset, range(idx_start *
        n_test, bound_test))
    return train_parition, test_parition
