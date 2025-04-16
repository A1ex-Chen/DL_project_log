def test_outputs_serialization(self):
    outputs_orig = CustomOutput(images=[PIL.Image.new('RGB', (4, 4))])
    serialized = pkl.dumps(outputs_orig)
    outputs_copy = pkl.loads(serialized)
    assert dir(outputs_orig) == dir(outputs_copy)
    assert dict(outputs_orig) == dict(outputs_copy)
    assert vars(outputs_orig) == vars(outputs_copy)
