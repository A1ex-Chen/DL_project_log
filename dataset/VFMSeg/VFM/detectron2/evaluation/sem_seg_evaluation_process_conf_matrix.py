def process_conf_matrix(self, n_cls, mat):
    acc = np.full(n_cls, np.nan, dtype=np.float)
    iou = np.full(n_cls, np.nan, dtype=np.float)
    tp = mat.diagonal()[:-1].astype(np.float)
    pos_gt = np.sum(mat[:-1, :-1], axis=0).astype(np.float)
    class_weights = pos_gt / np.sum(pos_gt)
    pos_pred = np.sum(mat[:-1, :-1], axis=1).astype(np.float)
    acc_valid = pos_gt > 0
    acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
    iou_valid = pos_gt + pos_pred > 0
    union = pos_gt + pos_pred - tp
    iou[acc_valid] = tp[acc_valid] / union[acc_valid]
    macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
    miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
    fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
    pacc = np.sum(tp) / np.sum(pos_gt)
    return macc, miou, fiou, pacc
