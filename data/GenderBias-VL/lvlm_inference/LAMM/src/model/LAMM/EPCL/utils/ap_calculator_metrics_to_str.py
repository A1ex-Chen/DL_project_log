def metrics_to_str(self, overall_ret, per_class=True):
    mAP_strs = []
    AR_strs = []
    per_class_metrics = []
    for ap_iou_thresh in self.ap_iou_thresh:
        mAP = overall_ret[ap_iou_thresh]['mAP'] * 100
        mAP_strs.append(f'{mAP:.2f}')
        ar = overall_ret[ap_iou_thresh]['AR'] * 100
        AR_strs.append(f'{ar:.2f}')
        if per_class:
            per_class_metrics.append('-' * 5)
            per_class_metrics.append(f'IOU Thresh={ap_iou_thresh}')
            for x in list(overall_ret[ap_iou_thresh].keys()):
                if x == 'mAP' or x == 'AR':
                    pass
                else:
                    met_str = f'{x}: {overall_ret[ap_iou_thresh][x] * 100:.2f}'
                    per_class_metrics.append(met_str)
    ap_header = [f'mAP{x:.2f}' for x in self.ap_iou_thresh]
    ap_str = ', '.join(ap_header)
    ap_str += ': ' + ', '.join(mAP_strs)
    ap_str += '\n'
    ar_header = [f'AR{x:.2f}' for x in self.ap_iou_thresh]
    ap_str += ', '.join(ar_header)
    ap_str += ': ' + ', '.join(AR_strs)
    if per_class:
        per_class_metrics = '\n'.join(per_class_metrics)
        ap_str += '\n'
        ap_str += per_class_metrics
    return ap_str
