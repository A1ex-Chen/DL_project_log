@contextlib.contextmanager
def mock_fastrcnn_outputs_inference(tensor_mode, check=True,
    box_predictor_type=FastRCNNOutputLayers):
    with mock.patch.object(box_predictor_type, 'inference', autospec=True,
        side_effect=Caffe2FastRCNNOutputsInference(tensor_mode)
        ) as mocked_func:
        yield
    if check:
        assert mocked_func.call_count > 0
