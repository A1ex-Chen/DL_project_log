def sample_2d_data(dset_id):
    assert dset_id in [1, 2]
    if dset_id == 1:
        dset_fn = sample_data_1_a
    else:
        dset_fn = sample_data_2_a
    train_data, test_data = dset_fn(10000), dset_fn(2500)
    return train_data.astype('float32'), test_data.astype('float32')
