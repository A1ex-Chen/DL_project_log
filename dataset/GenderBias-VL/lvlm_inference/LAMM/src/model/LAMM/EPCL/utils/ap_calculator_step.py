def step(self, predicted_box_corners, sem_cls_probs, objectness_probs,
    point_cloud, gt_box_corners, gt_box_sem_cls_labels, gt_box_present):
    """
        Perform NMS on predicted boxes and threshold them according to score.
        Convert GT boxes
        """
    gt_box_corners = gt_box_corners.cpu().detach().numpy()
    gt_box_sem_cls_labels = gt_box_sem_cls_labels.cpu().detach().numpy()
    gt_box_present = gt_box_present.cpu().detach().numpy()
    batch_gt_map_cls = self.make_gt_list(gt_box_corners,
        gt_box_sem_cls_labels, gt_box_present)
    batch_pred_map_cls = parse_predictions(predicted_box_corners,
        sem_cls_probs, objectness_probs, point_cloud, self.ap_config_dict)
    self.accumulate(batch_pred_map_cls, batch_gt_map_cls)
