@skip_mps
def test_attention_slicing_forward_pass(self):
    test_max_difference = torch_device == 'cpu'
    test_mean_pixel_difference = False
    self._test_attention_slicing_forward_pass(test_max_difference=
        test_max_difference, test_mean_pixel_difference=
        test_mean_pixel_difference)
