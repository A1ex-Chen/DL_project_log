def compute_ATE(gt, pred):
    """Compute RMSE of ATE
    Args:
        gt: ground-truth poses
        pred: predicted poses
    """
    errors = []
    for i in range(len(pred)):
        cur_gt = gt[i]
        gt_xyz = cur_gt[:3, 3]
        cur_pred = pred[i]
        pred_xyz = cur_pred[:3, 3]
        align_err = gt_xyz - pred_xyz
        errors.append(np.sqrt(np.sum(align_err ** 2)))
    ate = np.sqrt(np.mean(np.asarray(errors) ** 2))
    return ate
