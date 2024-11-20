def __init__(self, training_data=None, number_of_bins=100,
    significance_level=0.05, z_threshold=None, whitening=False, max_dims=
    None, cache_folder=None):
    """
        NDB Evaluation Class
        :param training_data: Optional - the training samples - array of m x d floats (m samples of dimension d)
        :param number_of_bins: Number of bins (clusters) default=100
        :param significance_level: The statistical significance level for the two-sample test
        :param z_threshold: Allow defining a threshold in terms of difference/SE for defining a bin as statistically different
        :param whitening: Perform data whitening - subtract mean and divide by per-dimension std
        :param max_dims: Max dimensions to use in K-means. By default derived automatically from d
        :param bins_file: Optional - file to write / read-from the clusters (to avoid re-calculation)
        """
    self.number_of_bins = number_of_bins
    self.significance_level = significance_level
    self.z_threshold = z_threshold
    self.whitening = whitening
    self.ndb_eps = 1e-06
    self.training_mean = 0.0
    self.training_std = 1.0
    self.max_dims = max_dims
    self.cache_folder = cache_folder
    self.bin_centers = None
    self.bin_proportions = None
    self.ref_sample_size = None
    self.used_d_indices = None
    self.results_file = None
    self.test_name = 'ndb_{}_bins_{}'.format(self.number_of_bins, 'whiten' if
        self.whitening else 'orig')
    self.cached_results = {}
    if self.cache_folder:
        self.results_file = os.path.join(cache_folder, self.test_name +
            '_results.pkl')
        if os.path.isfile(self.results_file):
            self.cached_results = pkl.load(open(self.results_file, 'rb'))
    if training_data is not None or cache_folder is not None:
        bins_file = None
        if cache_folder:
            os.makedirs(cache_folder, exist_ok=True)
            bins_file = os.path.join(cache_folder, self.test_name + '.pkl')
        self.construct_bins(training_data, bins_file)
