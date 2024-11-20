def process(self, inputs, outputs):
    for evaluator in self._evaluators:
        evaluator.process(inputs, outputs)
