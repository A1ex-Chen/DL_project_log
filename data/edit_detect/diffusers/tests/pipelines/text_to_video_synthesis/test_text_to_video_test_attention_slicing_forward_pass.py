@unittest.skipIf(torch_device != 'cuda', reason=
    "Feature isn't heavily used. Test in CUDA environment only.")
def test_attention_slicing_forward_pass(self):
    self._test_attention_slicing_forward_pass(test_mean_pixel_difference=
        False, expected_max_diff=0.003)
