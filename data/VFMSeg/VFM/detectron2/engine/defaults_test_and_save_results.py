def test_and_save_results():
    self._last_eval_results = self.test(self.cfg, self.model)
    return self._last_eval_results
