@unittest.skip(reason='bf16 not supported and requires CUDA')
def test_float16_inference(self):
    super().test_float16_inference()
