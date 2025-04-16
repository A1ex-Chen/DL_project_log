def process(self, inputs, outputs):
    """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
    for input, output in zip(inputs, outputs):
        output = output['sem_seg'].argmax(dim=0).to(self._cpu_device)
        pred = np.array(output, dtype=np.int)
        gt_filename = self.input_file_to_gt_file[input['file_name']]
        gt = self.sem_seg_loading_fn(gt_filename, dtype=np.int)
        if self.label_group:
            pred_remapped = self.remap(pred + 1)
            gt_remapped = self.remap(gt)
            gt_remapped[gt_remapped == self._ignore_label] = self.n_merged_cls
            self._conf_matrix_reduced += np.bincount((self.n_merged_cls + 1
                ) * pred_remapped.reshape(-1) + gt_remapped.reshape(-1),
                minlength=self._conf_matrix_reduced.size).reshape(self.
                _conf_matrix_reduced.shape)
        gt -= 1
        gt[gt == self._ignore_label - 1] = self._num_classes
        self._conf_matrix += np.bincount((self._num_classes + 1) * pred.
            reshape(-1) + gt.reshape(-1), minlength=self._conf_matrix.size
            ).reshape(self._conf_matrix.shape)
        self._predictions.extend(self.encode_json_sem_seg(pred, input[
            'file_name']))
