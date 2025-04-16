def evaluate(self, query_samples, model_label=None):
    """
        Assign each sample to the nearest bin center (in L2). Pre-whiten if required. and calculate the NDB
        (Number of statistically Different Bins) and JS divergence scores.
        :param query_samples: An array of m x d floats (m samples of dimension d)
        :param model_label: optional label string for the evaluated model, allows plotting results of multiple models
        :return: results dictionary containing NDB and JS scores and array of labels (assigned bin for each query sample)
        """
    n = query_samples.shape[0]
    query_bin_proportions, query_bin_assignments = (self.
        __calculate_bin_proportions(query_samples))
    different_bins = NDB.two_proportions_z_test(self.bin_proportions, self.
        ref_sample_size, query_bin_proportions, n, significance_level=self.
        significance_level, z_threshold=self.z_threshold)
    ndb = np.count_nonzero(different_bins)
    print('ndb', ndb)
    js = NDB.jensen_shannon_divergence(self.bin_proportions,
        query_bin_proportions)
    results = {'NDB': ndb, 'JS': js, 'Proportions': query_bin_proportions,
        'N': n, 'Bin-Assignment': query_bin_assignments, 'Different-Bins':
        different_bins}
    if model_label:
        print('Results for {} samples from {}: '.format(n, model_label), end=''
            )
        self.cached_results[model_label] = results
        if self.results_file:
            pkl.dump(self.cached_results, open(self.results_file, 'wb'))
    print('NDB =', ndb, 'NDB/K =', ndb / self.number_of_bins, ', JS =', js)
    return results
