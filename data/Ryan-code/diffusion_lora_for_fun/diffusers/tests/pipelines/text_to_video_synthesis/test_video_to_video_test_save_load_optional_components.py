@is_flaky()
def test_save_load_optional_components(self):
    super().test_save_load_optional_components(expected_max_difference=0.001)
