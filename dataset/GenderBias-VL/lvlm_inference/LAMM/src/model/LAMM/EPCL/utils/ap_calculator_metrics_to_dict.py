def metrics_to_dict(self, overall_ret):
    metrics_dict = {}
    for ap_iou_thresh in self.ap_iou_thresh:
        metrics_dict[f'mAP_{ap_iou_thresh}'] = overall_ret[ap_iou_thresh]['mAP'
            ] * 100
        metrics_dict[f'AR_{ap_iou_thresh}'] = overall_ret[ap_iou_thresh]['AR'
            ] * 100
    return metrics_dict
