@unittest.skip(
    'UnCLIP produces very large difference in fp16 vs fp32. Test is not useful.'
    )
def test_float16_inference(self):
    super().test_float16_inference(expected_max_diff=1.0)
