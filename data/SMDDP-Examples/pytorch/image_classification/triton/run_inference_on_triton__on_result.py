def _on_result(self, ids, x, y_real, output_names, result, error):
    with self._sync:
        if error:
            self._errors.append(error)
        else:
            y_pred = {name: result.as_numpy(name) for name in output_names}
            self._results.put((ids, x, y_pred, y_real))
        self._num_waiting_for -= 1
        self._sync.notify_all()
