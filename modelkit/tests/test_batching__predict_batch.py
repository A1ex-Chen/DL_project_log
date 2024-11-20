def _predict_batch(self, items):
    return [(item, position_in_batch, len(items)) for position_in_batch,
        item in enumerate(items)]
