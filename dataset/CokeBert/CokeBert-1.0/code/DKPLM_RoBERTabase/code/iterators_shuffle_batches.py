def shuffle_batches(batches, seed):
    with data_utils.numpy_seed(seed):
        np.random.shuffle(batches)
    return batches
