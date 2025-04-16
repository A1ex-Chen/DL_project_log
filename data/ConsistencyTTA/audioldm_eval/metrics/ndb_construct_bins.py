def construct_bins(self, training_samples, bins_file):
    """
        Performs K-means clustering of the training samples
        :param training_samples: An array of m x d floats (m samples of dimension d)
        """
    n, d = training_samples.shape
    k = self.number_of_bins
    if self.whitening:
        self.training_mean = np.mean(training_samples, axis=0)
        self.training_std = np.std(training_samples, axis=0) + self.ndb_eps
    if self.max_dims is None and d > 1000:
        self.max_dims = d // 6
    whitened_samples = (training_samples - self.training_mean
        ) / self.training_std
    d_used = d if self.max_dims is None else min(d, self.max_dims)
    self.used_d_indices = np.random.choice(d, d_used, replace=False)
    if n // k > 1000:
        print(
            'Training data size should be ~500 times the number of bins (for reasonable speed and accuracy)'
            )
    clusters = KMeans(n_clusters=k, max_iter=100).fit(whitened_samples[:,
        self.used_d_indices])
    bin_centers = np.zeros([k, d])
    for i in range(k):
        bin_centers[i, :] = np.mean(whitened_samples[clusters.labels_ == i,
            :], axis=0)
    label_vals, label_counts = np.unique(clusters.labels_, return_counts=True)
    bin_order = np.argsort(-label_counts)
    self.bin_proportions = label_counts[bin_order] / np.sum(label_counts)
    self.bin_centers = bin_centers[bin_order, :]
    self.ref_sample_size = n
    self.__write_to_bins_file(bins_file)
