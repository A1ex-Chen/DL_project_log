def __calculate_bin_proportions(self, samples):
    if self.bin_centers is None:
        print(
            'First run construct_bins on samples from the reference training data'
            )
    assert samples.shape[1] == self.bin_centers.shape[1]
    n, d = samples.shape
    k = self.bin_centers.shape[0]
    D = np.zeros([n, k], dtype=samples.dtype)
    whitened_samples = (samples - self.training_mean) / self.training_std
    for i in range(k):
        print('.', end='', flush=True)
        D[:, i] = np.linalg.norm(whitened_samples[:, self.used_d_indices] -
            self.bin_centers[i, self.used_d_indices], ord=2, axis=1)
    print()
    labels = np.argmin(D, axis=1)
    probs = np.zeros([k])
    label_vals, label_counts = np.unique(labels, return_counts=True)
    probs[label_vals] = label_counts / n
    return probs, labels
