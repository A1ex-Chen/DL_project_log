def _eval_predictions(self, predictions):
    """
        Evaluate predictions. Fill self._results with the metrics of the tasks.

        Args:
            predictions (list[dict]): list of outputs from the model
        """
    self._logger.info('Preparing results in the LVIS format ...')
    lvis_results = list(itertools.chain(*[x['instances'] for x in predictions])
        )
    tasks = self._tasks or self._tasks_from_predictions(lvis_results)
    if hasattr(self._metadata, 'thing_dataset_id_to_contiguous_id'):
        reverse_id_mapping = {v: k for k, v in self._metadata.
            thing_dataset_id_to_contiguous_id.items()}
        for result in lvis_results:
            result['category_id'] = reverse_id_mapping[result['category_id']]
    else:
        for result in lvis_results:
            result['category_id'] += 1
    if self._output_dir:
        file_path = os.path.join(self._output_dir,
            'lvis_instances_results.json')
        self._logger.info('Saving results to {}'.format(file_path))
        with PathManager.open(file_path, 'w') as f:
            f.write(json.dumps(lvis_results))
            f.flush()
    if not self._do_evaluation:
        self._logger.info('Annotations are not available for evaluation.')
        return
    self._logger.info('Evaluating predictions ...')
    for task in sorted(tasks):
        res = _evaluate_predictions_on_lvis(self._lvis_api, lvis_results,
            task, max_dets_per_image=self._max_dets_per_image, class_names=
            self._metadata.get('thing_classes'))
        self._results[task] = res
