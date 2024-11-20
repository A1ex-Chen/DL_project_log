def _generate_proposals(self, images, objectness_logits_pred,
    anchor_deltas_pred, gt_instances=None):
    assert isinstance(images, ImageList)
    if self.tensor_mode:
        im_info = images.image_sizes
    else:
        im_info = torch.tensor([[im_sz[0], im_sz[1], 1.0] for im_sz in
            images.image_sizes]).to(images.tensor.device)
    assert isinstance(im_info, torch.Tensor)
    rpn_rois_list = []
    rpn_roi_probs_list = []
    for scores, bbox_deltas, cell_anchors_tensor, feat_stride in zip(
        objectness_logits_pred, anchor_deltas_pred, iter(self.
        anchor_generator.cell_anchors), self.anchor_generator.strides):
        scores = scores.detach()
        bbox_deltas = bbox_deltas.detach()
        rpn_rois, rpn_roi_probs = torch.ops._caffe2.GenerateProposals(scores,
            bbox_deltas, im_info, cell_anchors_tensor, spatial_scale=1.0 /
            feat_stride, pre_nms_topN=self.pre_nms_topk[self.training],
            post_nms_topN=self.post_nms_topk[self.training], nms_thresh=
            self.nms_thresh, min_size=self.min_box_size, angle_bound_on=
            True, angle_bound_lo=-180, angle_bound_hi=180,
            clip_angle_thresh=1.0, legacy_plus_one=False)
        rpn_rois_list.append(rpn_rois)
        rpn_roi_probs_list.append(rpn_roi_probs)
    if len(objectness_logits_pred) == 1:
        rpn_rois = rpn_rois_list[0]
        rpn_roi_probs = rpn_roi_probs_list[0]
    else:
        assert len(rpn_rois_list) == len(rpn_roi_probs_list)
        rpn_post_nms_topN = self.post_nms_topk[self.training]
        device = rpn_rois_list[0].device
        input_list = [to_device(x, 'cpu') for x in rpn_rois_list +
            rpn_roi_probs_list]
        feature_strides = list(self.anchor_generator.strides)
        rpn_min_level = int(math.log2(feature_strides[0]))
        rpn_max_level = int(math.log2(feature_strides[-1]))
        assert rpn_max_level - rpn_min_level + 1 == len(rpn_rois_list
            ), 'CollectRpnProposals requires continuous levels'
        rpn_rois = torch.ops._caffe2.CollectRpnProposals(input_list,
            rpn_max_level=2 + len(rpn_rois_list) - 1, rpn_min_level=2,
            rpn_post_nms_topN=rpn_post_nms_topN)
        rpn_rois = to_device(rpn_rois, device)
        rpn_roi_probs = []
    proposals = self.c2_postprocess(im_info, rpn_rois, rpn_roi_probs, self.
        tensor_mode)
    return proposals, {}
