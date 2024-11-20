def _bucket_expected_max(bucket_times, ngpus):
    expected_max_arr = []
    for samples in bucket_times:
        expected_max = _bucket_estimate_max_expected(samples, ngpus)
        expected_max_arr.append(expected_max)
    return expected_max_arr
