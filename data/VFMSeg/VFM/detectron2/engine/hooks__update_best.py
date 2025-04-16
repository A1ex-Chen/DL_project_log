def _update_best(self, val, iteration):
    if math.isnan(val) or math.isinf(val):
        return False
    self.best_metric = val
    self.best_iter = iteration
    return True
