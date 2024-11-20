@skip_mps
def test_attention_slicing_forward_pass(self):
    return super().test_attention_slicing_forward_pass(expected_max_diff=0.005)
