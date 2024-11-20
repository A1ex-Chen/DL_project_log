def test_inference_batch_consistent(self, batch_sizes=[2]):
    self._test_inference_batch_consistent(batch_sizes=batch_sizes,
        batch_generator=False)
