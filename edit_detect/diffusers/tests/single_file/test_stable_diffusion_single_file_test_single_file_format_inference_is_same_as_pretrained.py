def test_single_file_format_inference_is_same_as_pretrained(self):
    super().test_single_file_format_inference_is_same_as_pretrained(
        expected_max_diff=0.001)
