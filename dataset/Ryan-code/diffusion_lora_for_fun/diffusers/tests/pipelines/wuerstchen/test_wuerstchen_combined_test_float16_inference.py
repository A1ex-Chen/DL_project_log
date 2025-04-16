@unittest.skip(reason='flakey and float16 requires CUDA')
def test_float16_inference(self):
    super().test_float16_inference()
