def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
    b = 0
    total_b = 0
    while 1:
        if b == 0:
            if seed is not None:
                np.random.seed(seed + total_b)
            if shuffle:
                index_array = np.random.permutation(N)
            else:
                index_array = np.arange(N)
        current_index = b * batch_size % N
        if N >= current_index + batch_size:
            current_batch_size = batch_size
        else:
            current_batch_size = N - current_index
        if current_batch_size == batch_size:
            b += 1
        else:
            b = 0
        total_b += 1
        yield index_array[current_index:current_index + current_batch_size
            ], current_index, current_batch_size
