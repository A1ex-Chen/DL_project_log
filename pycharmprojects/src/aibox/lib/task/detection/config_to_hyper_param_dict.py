def to_hyper_param_dict(self) ->Dict[str, Union[int, float, str, bool, Tensor]
    ]:
    hyper_param_dict = super().to_hyper_param_dict()
    hyper_param_dict.update({'algorithm_name': str(self.algorithm_name),
        'backbone_name': str(self.backbone_name), 'anchor_ratios': str(self
        .anchor_ratios), 'anchor_sizes': str(self.anchor_sizes),
        'backbone_pretrained': self.backbone_pretrained,
        'backbone_num_frozen_levels': self.backbone_num_frozen_levels,
        'train_rpn_pre_nms_top_n': self.train_rpn_pre_nms_top_n,
        'train_rpn_post_nms_top_n': self.train_rpn_post_nms_top_n,
        'eval_rpn_pre_nms_top_n': self.eval_rpn_pre_nms_top_n,
        'eval_rpn_post_nms_top_n': self.eval_rpn_post_nms_top_n,
        'num_anchor_samples_per_batch': self.num_anchor_samples_per_batch,
        'num_proposal_samples_per_batch': self.
        num_proposal_samples_per_batch, 'num_detections_per_image': self.
        num_detections_per_image, 'anchor_smooth_l1_loss_beta': self.
        anchor_smooth_l1_loss_beta, 'proposal_smooth_l1_loss_beta': self.
        proposal_smooth_l1_loss_beta, 'proposal_nms_threshold': self.
        proposal_nms_threshold, 'detection_nms_threshold': self.
        detection_nms_threshold, 'eval_quality': str(self.eval_quality)})
    return hyper_param_dict
