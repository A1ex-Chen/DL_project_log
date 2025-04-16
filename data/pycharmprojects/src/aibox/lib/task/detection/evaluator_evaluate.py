def evaluate(self, model: Model, returns_coco_result: bool=False) ->Evaluation:
    prediction = self.predict(model)
    evaluation = self._evaluate_with_condition(prediction, self._quality,
        self._size, returns_coco_result)
    return evaluation
