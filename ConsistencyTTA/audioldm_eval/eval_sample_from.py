def sample_from(self, samples, number_to_use):
    assert samples.shape[0] >= number_to_use
    rand_order = np.random.permutation(samples.shape[0])
    return samples[rand_order[:samples.shape[0]], :]
