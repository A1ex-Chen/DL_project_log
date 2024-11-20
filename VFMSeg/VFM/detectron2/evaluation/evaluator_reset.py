def reset(self):
    for evaluator in self._evaluators:
        evaluator.reset()
