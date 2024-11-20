@contextlib.contextmanager
def mock_mask_rcnn_inference(tensor_mode, patched_module, check=True):
    with mock.patch('{}.mask_rcnn_inference'.format(patched_module),
        side_effect=Caffe2MaskRCNNInference()) as mocked_func:
        yield
    if check:
        assert mocked_func.call_count > 0
