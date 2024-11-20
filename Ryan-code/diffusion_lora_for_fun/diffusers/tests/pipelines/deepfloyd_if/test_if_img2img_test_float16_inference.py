@unittest.skipIf(torch_device != 'cuda', reason='float16 requires CUDA')
def test_float16_inference(self):
    super().test_float16_inference(expected_max_diff=0.1)
