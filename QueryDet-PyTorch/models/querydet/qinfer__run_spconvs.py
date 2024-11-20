def _run_spconvs(self, x, filters):
    y = filters(x)
    return y.dense(channels_first=False)
