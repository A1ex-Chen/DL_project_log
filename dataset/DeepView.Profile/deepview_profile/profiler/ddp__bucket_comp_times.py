def _bucket_comp_times(path_to_file):
    data_matrix = []
    first_step, last_step = get_first_last_step(path_to_file)
    NUM_STEPS = 25
    forward_time_acc = 0
    for step in range(first_step + 1, first_step + NUM_STEPS + 1):
        fw_time, bucket_comp_times = get_ddp_forward_backward_times(
            path_to_file, step)
        forward_time_acc += fw_time
        """
        storing as:
        [bucket_0 time1, bucket_1 time1, ... , bucket_n time1]
        [bucket_0 time2, bucket_1 time2, ... , bucket_n time2]
        ...
        """
        data_matrix.append(bucket_comp_times)
    data_numpy = np.array(data_matrix)
    """
    store as :
    [bucket_0 time1, bucket_0 time2, ...., bucket_0 time n]
    [bucket_1 time1, bucket_1 time2, ...., bucket_1 time n]
    """
    data_transpose = np.transpose(data_numpy)
    return forward_time_acc / NUM_STEPS, data_transpose
