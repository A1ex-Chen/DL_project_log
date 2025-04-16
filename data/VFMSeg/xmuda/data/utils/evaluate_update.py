def update(self, pred_label, gt_label):
    """Update per instance

        Args:
            pred_label (np.ndarray): (num_points)
            gt_label (np.ndarray): (num_points,)

        """
    gt_label[gt_label == -100] = self.num_classes
    confusion_matrix = CM(gt_label.flatten(), pred_label.flatten(), labels=
        self.labels)
    self.confusion_matrix += confusion_matrix
