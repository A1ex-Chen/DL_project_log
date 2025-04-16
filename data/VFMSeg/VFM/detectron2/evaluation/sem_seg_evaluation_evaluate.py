def evaluate(self):
    """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
    if self._distributed:
        synchronize()
        conf_matrix_list = all_gather(self._conf_matrix)
        conf_matrix_list_reduced = all_gather(self._conf_matrix_reduced)
        self._predictions = all_gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not is_main_process():
            return
        self._conf_matrix = np.zeros_like(self._conf_matrix)
        self._conf_matrix_reduced = np.zeros_like(self._conf_matrix_reduced)
        for conf_matrix in conf_matrix_list:
            self._conf_matrix += conf_matrix
        if self.label_group:
            for conf_matrix in conf_matrix_list_reduced:
                self._conf_matrix_reduced += conf_matrix
    if self._output_dir:
        PathManager.mkdirs(self._output_dir)
        file_path = os.path.join(self._output_dir, 'sem_seg_predictions.json')
        with PathManager.open(file_path, 'w') as f:
            f.write(json.dumps(self._predictions))
    acc = np.full(self._num_classes, np.nan, dtype=np.float)
    iou = np.full(self._num_classes, np.nan, dtype=np.float)
    tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
    pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
    class_weights = pos_gt / np.sum(pos_gt)
    pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
    acc_valid = pos_gt > 0
    acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
    iou_valid = pos_gt + pos_pred > 0
    union = pos_gt + pos_pred - tp
    iou[acc_valid] = tp[acc_valid] / union[acc_valid]
    macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
    miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
    fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
    pacc = np.sum(tp) / np.sum(pos_gt)
    res = {}
    res['mIoU'] = 100 * miou
    if self.label_group:
        _, miou_p, _, _ = self.process_conf_matrix(self.n_merged_cls, self.
            _conf_matrix_reduced)
        res['mIoU_Parts'] = 100 * miou_p
    res['fwIoU'] = 100 * fiou
    for i, name in enumerate(self._class_names):
        res['IoU-{}'.format(name)] = 100 * iou[i]
    res['mACC'] = 100 * macc
    res['pACC'] = 100 * pacc
    for i, name in enumerate(self._class_names):
        res['ACC-{}'.format(name)] = 100 * acc[i]
    if self._output_dir:
        file_path = os.path.join(self._output_dir, 'sem_seg_evaluation.pth')
        with PathManager.open(file_path, 'wb') as f:
            torch.save(res, f)
    results = OrderedDict({'sem_seg': res})
    self._logger.info(results)
    return results
