def _print_biases(self):
    m = self.model[-1]
    for mi in m.m:
        b = mi.bias.detach().view(m.na, -1).T
        LOGGER.info(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[
            1], *b[:5].mean(1).tolist(), b[5:].mean()))
