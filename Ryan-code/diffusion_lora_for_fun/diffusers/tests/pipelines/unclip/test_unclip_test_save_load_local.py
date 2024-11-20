@skip_mps
def test_save_load_local(self):
    return super().test_save_load_local(expected_max_difference=0.005)
