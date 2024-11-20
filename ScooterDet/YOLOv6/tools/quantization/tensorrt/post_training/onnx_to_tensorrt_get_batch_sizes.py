def get_batch_sizes(max_batch_size):
    max_exponent = math.log2(max_batch_size)
    for i in range(int(max_exponent) + 1):
        batch_size = 2 ** i
        yield batch_size
    if max_batch_size != batch_size:
        yield max_batch_size
