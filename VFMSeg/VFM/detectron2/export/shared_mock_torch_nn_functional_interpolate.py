@contextlib.contextmanager
def mock_torch_nn_functional_interpolate():
    if torch.onnx.is_in_onnx_export():
        with mock.patch('torch.nn.functional.interpolate', side_effect=
            onnx_compatibale_interpolate):
            yield
    else:
        yield
