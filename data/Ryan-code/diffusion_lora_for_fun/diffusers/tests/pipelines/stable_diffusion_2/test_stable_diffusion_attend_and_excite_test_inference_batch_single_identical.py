def test_inference_batch_single_identical(self):
    self._test_inference_batch_single_identical(batch_size=2,
        expected_max_diff=0.0007)
