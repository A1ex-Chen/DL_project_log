def tp_fp(self):
    tp = self.matrix.diagonal()
    fp = self.matrix.sum(1) - tp
    return tp[:-1], fp[:-1]
