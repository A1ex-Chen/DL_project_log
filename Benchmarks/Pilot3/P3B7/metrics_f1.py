def f1(self, y_hat, y):
    """Get the batch F1 score"""
    scores = {}
    for task, pred in y_hat.items():
        scores[task] = self.metrics[task](pred, y[task])
    return scores
