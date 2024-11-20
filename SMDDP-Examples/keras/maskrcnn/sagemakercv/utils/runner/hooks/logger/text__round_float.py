def _round_float(self, items):
    if isinstance(items, list):
        return [self._round_float(item) for item in items]
    elif isinstance(items, float):
        return round(items, 5)
    else:
        return items
