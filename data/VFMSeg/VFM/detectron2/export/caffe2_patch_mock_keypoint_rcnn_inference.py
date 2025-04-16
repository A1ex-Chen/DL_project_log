@contextlib.contextmanager
def mock_keypoint_rcnn_inference(tensor_mode, patched_module,
    use_heatmap_max_keypoint, check=True):
    with mock.patch('{}.keypoint_rcnn_inference'.format(patched_module),
        side_effect=Caffe2KeypointRCNNInference(use_heatmap_max_keypoint)
        ) as mocked_func:
        yield
    if check:
        assert mocked_func.call_count > 0
