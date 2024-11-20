def test_find_backend(self):
    simple_backend = find_backend('    if not is_torch_available():')
    self.assertEqual(simple_backend, 'torch')
    double_backend = find_backend(
        '    if not (is_torch_available() and is_transformers_available()):')
    self.assertEqual(double_backend, 'torch_and_transformers')
    triple_backend = find_backend(
        '    if not (is_torch_available() and is_transformers_available() and is_onnx_available()):'
        )
    self.assertEqual(triple_backend, 'torch_and_transformers_and_onnx')
