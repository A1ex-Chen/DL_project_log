def evaluate(self):
    results = OrderedDict()
    for evaluator in self._evaluators:
        result = evaluator.evaluate()
        if is_main_process() and result is not None:
            for k, v in result.items():
                assert k not in results, 'Different evaluators produce results with the same key {}'.format(
                    k)
                results[k] = v
    return results
