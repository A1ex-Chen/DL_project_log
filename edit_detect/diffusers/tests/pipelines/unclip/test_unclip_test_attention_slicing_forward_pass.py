@skip_mps
def test_attention_slicing_forward_pass(self):
    test_max_difference = torch_device == 'cpu'
    self._test_attention_slicing_forward_pass(test_max_difference=
        test_max_difference, expected_max_diff=0.01)
