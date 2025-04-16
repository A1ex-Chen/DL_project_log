def _bucket_estimate_max_expected(bucket_times, ngpu):
    m = 1000
    np_samples = np.array(bucket_times)
    kde_samples = gaussian_kde(np_samples)
    z_arr = []
    for _ in range(m):
        num_resamples = kde_samples.resample(ngpu)
        z_arr.append(np.max(num_resamples))
    expected_max = np.mean(z_arr)
    return expected_max
