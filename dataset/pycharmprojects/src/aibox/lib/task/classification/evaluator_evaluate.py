def evaluate(self, model: Model) ->Evaluation:
    prediction = self.predict(model)
    evaluation = self._evaluate(prediction)
    return evaluation
