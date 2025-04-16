def evaluate(self):
    comm.synchronize()
    self._predictions = comm.gather(self._predictions)
    self._predictions = list(itertools.chain(*self._predictions))
    if not comm.is_main_process():
        return
    gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
    gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)
    with tempfile.TemporaryDirectory(prefix='panoptic_eval') as pred_dir:
        logger.info('Writing all panoptic predictions to {} ...'.format(
            pred_dir))
        for p in self._predictions:
            with open(os.path.join(pred_dir, p['file_name']), 'wb') as f:
                f.write(p.pop('png_string'))
        with open(gt_json, 'r') as f:
            json_data = json.load(f)
        json_data['annotations'] = self._predictions
        output_dir = self._output_dir or pred_dir
        predictions_json = os.path.join(output_dir, 'predictions.json')
        with PathManager.open(predictions_json, 'w') as f:
            f.write(json.dumps(json_data))
        with contextlib.redirect_stdout(io.StringIO()):
            pq_res = pq_compute(gt_json, PathManager.get_local_path(
                predictions_json), gt_folder=gt_folder, pred_folder=pred_dir)
    res = {}
    res['PQ'] = 100 * pq_res['All']['pq']
    res['SQ'] = 100 * pq_res['All']['sq']
    res['RQ'] = 100 * pq_res['All']['rq']
    res['PQ_th'] = 100 * pq_res['Things']['pq']
    res['SQ_th'] = 100 * pq_res['Things']['sq']
    res['RQ_th'] = 100 * pq_res['Things']['rq']
    res['PQ_st'] = 100 * pq_res['Stuff']['pq']
    res['SQ_st'] = 100 * pq_res['Stuff']['sq']
    res['RQ_st'] = 100 * pq_res['Stuff']['rq']
    results = OrderedDict({'panoptic_seg': res})
    _print_panoptic_results(pq_res)
    return results
