def test_correct_modelcard_format(self):
    assert not self.modelcard_has_connected_pipeline(
        'hf-internal-testing/tiny-random-kandinsky-v22-prior')
    assert self.modelcard_has_connected_pipeline(
        'hf-internal-testing/tiny-random-kandinsky-v22-decoder')
