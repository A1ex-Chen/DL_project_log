def step_meter(self, outputs, targets):
    if 'outputs' in outputs:
        outputs = outputs['outputs']
    self.step(predicted_box_corners=outputs['box_corners'], sem_cls_probs=
        outputs['sem_cls_prob'], objectness_probs=outputs['objectness_prob'
        ], point_cloud=targets['point_clouds'], gt_box_corners=targets[
        'gt_box_corners'], gt_box_sem_cls_labels=targets[
        'gt_box_sem_cls_label'], gt_box_present=targets['gt_box_present'])
