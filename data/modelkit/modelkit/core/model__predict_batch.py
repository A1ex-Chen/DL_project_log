@not_overriden
def _predict_batch(self, items: List[ItemType], **kwargs) ->List[ReturnType]:
    return [self._predict(p, **kwargs) for p in items]
