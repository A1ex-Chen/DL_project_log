def _eval_predictions(self, predictions, img_ids=None):
    """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
    self._logger.info('Preparing results for COCO format ...')
    coco_results = list(itertools.chain(*[x['instances'] for x in predictions])
        )
    if hasattr(self._metadata, 'thing_dataset_id_to_contiguous_id'):
        reverse_id_mapping = {v: k for k, v in self._metadata.
            thing_dataset_id_to_contiguous_id.items()}
        for result in coco_results:
            result['category_id'] = reverse_id_mapping[result['category_id']]
    if self._output_dir:
        file_path = os.path.join(self._output_dir,
            'coco_instances_results.json')
        self._logger.info('Saving results to {}'.format(file_path))
        with PathManager.open(file_path, 'w') as f:
            f.write(json.dumps(coco_results))
            f.flush()
    if not self._do_evaluation:
        self._logger.info('Annotations are not available for evaluation.')
        return
    self._logger.info('Evaluating predictions ...')
    assert self._tasks is None or set(self._tasks) == {'bbox'
        }, '[RotatedCOCOEvaluator] Only bbox evaluation is supported'
    coco_eval = self._evaluate_predictions_on_coco(self._coco_api, coco_results
        ) if len(coco_results) > 0 else None
    task = 'bbox'
    res = self._derive_coco_results(coco_eval, task, class_names=self.
        _metadata.get('thing_classes'))
    self._results[task] = res
