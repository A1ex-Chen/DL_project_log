def __read_from_bins_file(self, bins_file):
    if bins_file and os.path.isfile(bins_file):
        print('Loading binning results from', bins_file)
        bins_data = pkl.load(open(bins_file, 'rb'))
        self.bin_proportions = bins_data['proportions']
        self.bin_centers = bins_data['centers']
        self.ref_sample_size = bins_data['n']
        self.training_mean = bins_data['mean']
        self.training_std = bins_data['std']
        self.used_d_indices = bins_data['d_indices']
        return True
    return False
