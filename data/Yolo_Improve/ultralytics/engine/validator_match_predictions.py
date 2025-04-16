def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
    """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
    correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool
        )
    correct_class = true_classes[:, None] == pred_classes
    iou = iou * correct_class
    iou = iou.cpu().numpy()
    for i, threshold in enumerate(self.iouv.cpu().tolist()):
        if use_scipy:
            import scipy
            cost_matrix = iou * (iou >= threshold)
            if cost_matrix.any():
                labels_idx, detections_idx = (scipy.optimize.
                    linear_sum_assignment(cost_matrix, maximize=True))
                valid = cost_matrix[labels_idx, detections_idx] > 0
                if valid.any():
                    correct[detections_idx[valid], i] = True
        else:
            matches = np.nonzero(iou >= threshold)
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].
                        argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index
                        =True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index
                        =True)[1]]
                correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)
