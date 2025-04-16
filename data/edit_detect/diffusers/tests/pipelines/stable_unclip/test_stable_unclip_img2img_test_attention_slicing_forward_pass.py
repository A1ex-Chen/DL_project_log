def test_attention_slicing_forward_pass(self):
    test_max_difference = torch_device in ['cpu', 'mps']
    self._test_attention_slicing_forward_pass(test_max_difference=
        test_max_difference)
