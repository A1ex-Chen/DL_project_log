def _predict(self, item: bool, force_true=False, **_) ->bool:
    if force_true:
        return True
    return item
