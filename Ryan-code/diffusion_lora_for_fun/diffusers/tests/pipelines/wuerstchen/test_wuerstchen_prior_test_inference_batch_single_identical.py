@skip_mps
def test_inference_batch_single_identical(self):
    self._test_inference_batch_single_identical(expected_max_diff=0.3)
