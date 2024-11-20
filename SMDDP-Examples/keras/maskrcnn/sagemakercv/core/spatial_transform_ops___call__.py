def __call__(self, fpn_feats, rpn_box_rois):
    roi_features = multilevel_crop_and_resize(fpn_feats, rpn_box_rois,
        output_size=self.output_size, is_gpu_inference=self.is_gpu_inference)
    return roi_features
