def forward(self, x, box_lists):
    assert not self.training
    pooler_fmt_boxes = self.c2_preprocess(box_lists)
    num_level_assignments = len(self.level_poolers)
    if num_level_assignments == 1:
        if isinstance(self.level_poolers[0], ROIAlignRotated):
            c2_roi_align = torch.ops._caffe2.RoIAlignRotated
            aligned = True
        else:
            c2_roi_align = torch.ops._caffe2.RoIAlign
            aligned = self.level_poolers[0].aligned
        x0 = x[0]
        if x0.is_quantized:
            x0 = x0.dequantize()
        out = c2_roi_align(x0, pooler_fmt_boxes, order='NCHW',
            spatial_scale=float(self.level_poolers[0].spatial_scale),
            pooled_h=int(self.output_size[0]), pooled_w=int(self.
            output_size[1]), sampling_ratio=int(self.level_poolers[0].
            sampling_ratio), aligned=aligned)
        return out
    device = pooler_fmt_boxes.device
    assert self.max_level - self.min_level + 1 == 4, 'Currently DistributeFpnProposals only support 4 levels'
    fpn_outputs = torch.ops._caffe2.DistributeFpnProposals(to_device(
        pooler_fmt_boxes, 'cpu'), roi_canonical_scale=self.
        canonical_box_size, roi_canonical_level=self.canonical_level,
        roi_max_level=self.max_level, roi_min_level=self.min_level,
        legacy_plus_one=False)
    fpn_outputs = [to_device(x, device) for x in fpn_outputs]
    rois_fpn_list = fpn_outputs[:-1]
    rois_idx_restore_int32 = fpn_outputs[-1]
    roi_feat_fpn_list = []
    for roi_fpn, x_level, pooler in zip(rois_fpn_list, x, self.level_poolers):
        if isinstance(pooler, ROIAlignRotated):
            c2_roi_align = torch.ops._caffe2.RoIAlignRotated
            aligned = True
        else:
            c2_roi_align = torch.ops._caffe2.RoIAlign
            aligned = bool(pooler.aligned)
        if x_level.is_quantized:
            x_level = x_level.dequantize()
        roi_feat_fpn = c2_roi_align(x_level, roi_fpn, order='NCHW',
            spatial_scale=float(pooler.spatial_scale), pooled_h=int(self.
            output_size[0]), pooled_w=int(self.output_size[1]),
            sampling_ratio=int(pooler.sampling_ratio), aligned=aligned)
        roi_feat_fpn_list.append(roi_feat_fpn)
    roi_feat_shuffled = cat(roi_feat_fpn_list, dim=0)
    assert roi_feat_shuffled.numel() > 0 and rois_idx_restore_int32.numel(
        ) > 0, 'Caffe2 export requires tracing with a model checkpoint + input that can produce valid detections. But no detections were obtained with the given checkpoint and input!'
    roi_feat = torch.ops._caffe2.BatchPermutation(roi_feat_shuffled,
        rois_idx_restore_int32)
    return roi_feat
