def compute_miou(self):
    cm = self.overall_confusion_matrix.copy()
    gt_set = cm.sum(axis=1)
    all_set = cm.sum(axis=None)
    print(f'all_set:{all_set}')
    pred_set = cm.sum(axis=0)
    intersection = np.diag(cm)
    intersect = intersection.sum(axis=None)
    print(f'intersect:{intersect}')
    union = gt_set + pred_set - intersection
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = intersection / union.astype(np.float32)
        pacc = intersect / all_set.astype(np.float32)
        pacc = np.nan_to_num(pacc)
        iou = np.nan_to_num(iou)
        macc = intersection / gt_set.astype(np.float32)
        macc = np.nan_to_num(macc)
    miou = np.mean(iou)
    macc = np.mean(macc)
    return miou, pacc, macc, iou
