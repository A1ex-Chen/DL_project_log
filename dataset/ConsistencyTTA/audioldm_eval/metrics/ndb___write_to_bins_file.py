def __write_to_bins_file(self, bins_file):
    if bins_file:
        print('Caching binning results to', bins_file)
        bins_data = {'proportions': self.bin_proportions, 'centers': self.
            bin_centers, 'n': self.ref_sample_size, 'mean': self.
            training_mean, 'std': self.training_std, 'd_indices': self.
            used_d_indices}
        pkl.dump(bins_data, open(bins_file, 'wb'))
