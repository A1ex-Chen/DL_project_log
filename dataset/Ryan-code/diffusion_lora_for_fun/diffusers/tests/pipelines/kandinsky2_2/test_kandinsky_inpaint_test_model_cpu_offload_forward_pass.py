@is_flaky()
def test_model_cpu_offload_forward_pass(self):
    super().test_inference_batch_single_identical(expected_max_diff=0.0008)
