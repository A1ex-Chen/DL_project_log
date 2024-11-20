def _eval_predictions(self, predictions, img_ids=None):
    """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
    self._logger.info('Preparing results for COCO format ...')
    coco_results = list(itertools.chain(*[x['instances'] for x in predictions])
        )
    tasks = self._tasks or self._tasks_from_predictions(coco_results)
    if hasattr(self._metadata, 'thing_dataset_id_to_contiguous_id'):
        dataset_id_to_contiguous_id = (self._metadata.
            thing_dataset_id_to_contiguous_id)
        reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.
            items()}
        for result in coco_results:
            category_id = result['category_id']
            assert category_id in reverse_id_mapping, f'A prediction has class={category_id}, but the dataset only has class ids in {dataset_id_to_contiguous_id}.'
            result['category_id'] = reverse_id_mapping[category_id]
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
    self._logger.info('Evaluating predictions with {} COCO API...'.format(
        'unofficial' if self._use_fast_impl else 'official'))
    for task in sorted(tasks):
        assert task in {'bbox', 'segm', 'keypoints'
            }, f'Got unknown task: {task}!'
        coco_eval = _evaluate_predictions_on_coco(self._coco_api,
            coco_results, task, kpt_oks_sigmas=self._kpt_oks_sigmas,
            img_ids=img_ids, max_dets_per_image=self._max_dets_per_image
            ) if len(coco_results) > 0 else None
        res = self._derive_coco_results(coco_eval, task, class_names=self.
            _metadata.get('thing_classes'))
        self._results[task] = res
