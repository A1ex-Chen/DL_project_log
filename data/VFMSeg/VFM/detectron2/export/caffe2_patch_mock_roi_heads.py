@contextlib.contextmanager
def mock_roi_heads(self, tensor_mode=True):
    """
        Patching several inference functions inside ROIHeads and its subclasses

        Args:
            tensor_mode (bool): whether the inputs/outputs are caffe2's tensor
                format or not. Default to True.
        """
    kpt_heads_mod = keypoint_head.BaseKeypointRCNNHead.__module__
    mask_head_mod = mask_head.BaseMaskRCNNHead.__module__
    mock_ctx_managers = [mock_fastrcnn_outputs_inference(tensor_mode=
        tensor_mode, check=True, box_predictor_type=type(self.heads.
        box_predictor))]
    if getattr(self.heads, 'keypoint_on', False):
        mock_ctx_managers += [mock_keypoint_rcnn_inference(tensor_mode,
            kpt_heads_mod, self.use_heatmap_max_keypoint)]
    if getattr(self.heads, 'mask_on', False):
        mock_ctx_managers += [mock_mask_rcnn_inference(tensor_mode,
            mask_head_mod)]
    with contextlib.ExitStack() as stack:
        for mgr in mock_ctx_managers:
            stack.enter_context(mgr)
        yield
