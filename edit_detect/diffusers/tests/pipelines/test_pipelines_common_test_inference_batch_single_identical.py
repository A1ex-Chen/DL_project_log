def test_inference_batch_single_identical(self, batch_size=3,
    expected_max_diff=0.0001):
    self._test_inference_batch_single_identical(batch_size=batch_size,
        expected_max_diff=expected_max_diff)
