def train(self):
    """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
    super().train(self.start_iter, self.max_iter)
    if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
        assert hasattr(self, '_last_eval_results'
            ), 'No evaluation results obtained during training!'
        verify_results(self.cfg, self._last_eval_results)
        return self._last_eval_results
