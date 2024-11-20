def evaluate(self):
    if self._distributed:
        comm.synchronize()
        predictions = comm.gather(self._predictions, dst=0)
        predictions = list(itertools.chain(*predictions))
        if not comm.is_main_process():
            return {}
    else:
        predictions = self._predictions
    if len(predictions) == 0:
        return {}
    det_preds = []
    for pred in predictions:
        det_preds = det_preds + pred['instances']
    with open(self._out_json, 'w') as f:
        f.write(json.dumps(det_preds))
        f.flush()
    return {}
