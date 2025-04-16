def __call__(self, pred_keypoint_logits, pred_instances):
    output = alias(pred_keypoint_logits, 'kps_score')
    if all(isinstance(x, InstancesList) for x in pred_instances):
        assert len(pred_instances) == 1
        if self.use_heatmap_max_keypoint:
            device = output.device
            output = torch.ops._caffe2.HeatmapMaxKeypoint(to_device(output,
                'cpu'), pred_instances[0].pred_boxes.tensor,
                should_output_softmax=True)
            output = to_device(output, device)
            output = alias(output, 'keypoints_out')
        pred_instances[0].pred_keypoints = output
    return pred_keypoint_logits
