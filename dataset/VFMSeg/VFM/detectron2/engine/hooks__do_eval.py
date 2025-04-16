def _do_eval(self):
    results = self._func()
    if results:
        assert isinstance(results, dict
            ), 'Eval function must return a dict. Got {} instead.'.format(
            results)
        flattened_results = flatten_results_dict(results)
        for k, v in flattened_results.items():
            try:
                v = float(v)
            except Exception as e:
                raise ValueError(
                    "[EvalHook] eval_function should return a nested dict of float. Got '{}: {}' instead."
                    .format(k, v)) from e
        self.trainer.storage.put_scalars(**flattened_results,
            smoothing_hint=False)
    comm.synchronize()
