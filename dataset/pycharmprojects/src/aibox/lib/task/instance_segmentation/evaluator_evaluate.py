def evaluate(self, model: Model) ->Evaluation:
    prediction = self.predict(model, needs_inv_process=False)
    evaluation = self._evaluate_with_condition(prediction, self._quality,
        self._size, pred_needs_inv_process=True)
    return evaluation
