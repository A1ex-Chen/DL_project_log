def test_model_cpu_offload_forward_pass(self):
    super().test_model_cpu_offload_forward_pass(expected_max_diff=0.0005)
