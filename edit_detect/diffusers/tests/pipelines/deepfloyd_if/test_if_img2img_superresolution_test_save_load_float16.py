@unittest.skipIf(torch_device != 'cuda', reason='float16 requires CUDA')
def test_save_load_float16(self):
    super().test_save_load_float16(expected_max_diff=0.1)
