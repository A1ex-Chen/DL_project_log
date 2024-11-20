def tp_fp(self):
    """Returns true positives and false positives."""
    tp = self.matrix.diagonal()
    fp = self.matrix.sum(1) - tp
    return (tp[:-1], fp[:-1]) if self.task == 'detect' else (tp, fp)
