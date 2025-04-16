@unittest.skipIf(torch_device != 'cuda' or not is_xformers_available(),
    reason=
    'XFormers attention is only available with CUDA and `xformers` installed')
def test_xformers_attention_forwardGenerator_pass(self):
    self._test_xformers_attention_forwardGenerator_pass(expected_max_diff=0.001
        )
