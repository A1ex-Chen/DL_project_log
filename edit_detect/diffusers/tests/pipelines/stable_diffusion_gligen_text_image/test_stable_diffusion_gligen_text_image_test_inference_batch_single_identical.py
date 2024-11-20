def test_inference_batch_single_identical(self):
    super().test_inference_batch_single_identical(batch_size=3,
        expected_max_diff=0.003)
